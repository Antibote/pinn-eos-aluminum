"""PINN: физические штрафы, обучение и проверка физики для ``EOSNet``.

Производные берутся по логарифмам: ln ρ и ln T. Так все штрафы безразмерны и
одного порядка во всём диапазоне (ρ — 7 порядков, T — 8, P — 12).

Ограничения (их ``physics_terms`` возвращает как невязки, 0 — выполнено):
- согласованность (∂e/∂ρ)_T = (P − T(∂P/∂T)_ρ) / ρ²; в лог-переменных
  A = B, где A = ρ ∂e/∂lnρ / P, B = 1 − g, g = ∂lnP/∂lnT; невязка (A − B) / (|A| + 1 + |g|)
  ограничена |r| ≤ 1 (иначе её квадрат на необученной сети перебивает все остальные слагаемые);
- механическая устойчивость (∂P/∂ρ)_T > 0  ⇔  ∂lnP/∂lnρ > 0;
- термическая устойчивость (∂e/∂T)_ρ ≥ 0 (нестрого: в таблице есть плато при T ≲ 7 K);
- скорость звука c_s² = ∂P/∂ρ + T(∂P/∂T)² / (ρ² ∂e/∂T), 0 < c_s < c (c — скорость света);
  в штрафе «sound» неравенство умножено на ρ² ∂e/∂T, чтобы не делить на ∂e/∂T ≈ 0.

Таблица сама не вполне согласована: в холодной плотной части ρ²∂e/∂ρ ≈ 0.5 (P − T∂P/∂T).
Поэтому вес штрафа согласованности подбирается перебором (компромисс с точностью).
"""

from __future__ import annotations

import copy
import math
import time

import numpy as np
import torch

from eos.model import EOSNet, sample_collocation
from eos.metrics import C_LIGHT_KMS  # скорость света, км/с

LN10 = math.log(10.0)


def log_derivatives(model: EOSNet, rho: torch.Tensor, T: torch.Tensor,
                    create_graph: bool = True) -> dict[str, torch.Tensor]:
    """P, e и производные по ln ρ, ln T через autograd.

    Args:
        model: ``EOSNet``.
        rho: плотность, г/см³, форма (N,).
        T: температура, K, форма (N,).
        create_graph: True — производные можно использовать в loss.

    Returns:
        Словарь: P [ГПа], e [кДж/г], a = ∂lnP/∂lnρ, g = ∂lnP/∂lnT,
        e_r = ∂e/∂lnρ [кДж/г], e_t = ∂e/∂lnT [кДж/г].
    """
    lr = torch.log(rho).detach().requires_grad_(True)
    lt = torch.log(T).detach().requires_grad_(True)
    n = model.norm
    y = model.forward_norm(torch.exp(lr), torch.exp(lt))
    c = torch.as_tensor(n.y_center, dtype=y.dtype, device=y.device)
    s = torch.as_tensor(n.y_scale, dtype=y.dtype, device=y.device)
    raw = y * s + c
    log10P, e = raw[:, 0], n.E0 * torch.sinh(raw[:, 1])
    a, g = torch.autograd.grad(log10P.sum(), (lr, lt), create_graph=create_graph, retain_graph=True)
    e_r, e_t = torch.autograd.grad(e.sum(), (lr, lt), create_graph=create_graph)
    return {"P": 10 ** log10P, "e": e, "a": a * LN10, "g": g * LN10, "e_r": e_r, "e_t": e_t}


def physics_terms(d: dict[str, torch.Tensor], rho: torch.Tensor, E0: float = 1.0) -> dict[str, torch.Tensor]:
    """Безразмерные невязки ограничений по результату ``log_derivatives``.

    Returns:
        cons — относительная невязка согласованности, |cons| ≤ 1;
        mono_rho — max(0, −∂lnP/∂lnρ); mono_T — max(0, −∂e/∂lnT) / (|e| + E0);
        sound — нарушение 0 < c_s² < c², записанное без деления на ∂e/∂T:
            при ∂e/∂T ≥ 0 (его обеспечивает mono_T) это 0 < N < c² ρ² ∂e/∂lnT,
            N = c_s² · ρ² ∂e/∂lnT = P a ρ ∂e/∂lnT + (P g)²; нарушение делится на сумму модулей
            слагаемых, поэтому sound ∈ [0, 1) и не взрывается там, где ∂e/∂T ≈ 0.
    """
    P, a, g, e_t = d["P"], d["a"], d["g"], d["e_t"]
    A = rho * d["e_r"] / P
    cons = (A - 1 + g) / (A.abs() + 1 + g.abs())
    mono_rho = torch.relu(-a)
    mono_T = torch.relu(-e_t) / (d["e"].abs() + E0)
    # во float64: P², c²ρ²∂e/∂lnT доходят до 1e24 и больше — во float32 возможно ∞/∞ = NaN
    P64, a64, g64, e64, r64 = (x.double() for x in (P, a, g, e_t, rho))
    term_rho, term_T = P64 * a64 * r64 * e64, (P64 * g64) ** 2
    light = C_LIGHT_KMS ** 2 * r64 ** 2 * e64
    N = term_rho + term_T
    scale = term_rho.abs() + term_T + light.abs() + 1e-300
    sound = ((torch.relu(-N) + torch.relu(N - light)) / scale).to(P.dtype)
    return {"cons": cons, "mono_rho": mono_rho, "mono_T": mono_T, "sound": sound}


def sound_speed2(d: dict[str, torch.Tensor], rho: torch.Tensor) -> torch.Tensor:
    """c_s² [кДж/г = (км/с)²] = P a / ρ + (P g)² / (ρ² ∂e/∂lnT).

    |∂e/∂lnT| < 1e-12 (плато таблицы) заменяется на +1e-12, чтобы не было деления на ноль.
    """
    P, e_t = d["P"], d["e_t"]
    e_t = torch.where(e_t.abs() < 1e-12, torch.full_like(e_t, 1e-12), e_t)
    return P * d["a"] / rho + (P * d["g"]) ** 2 / (rho ** 2 * e_t)


def _data_loss_fn(err: torch.Tensor, kind: str, delta: float = 0.01) -> torch.Tensor:
    """Поэлементный loss по ошибке err в нормированном пространстве.

    mse — квадрат; l1 — модуль; huber — квадрат при |err| < delta, модуль дальше;
    logcosh — delta² · log cosh(err / delta): гладкий аналог huber.
    """
    if kind == "mse":
        return err ** 2
    if kind == "l1":
        return err.abs()
    if kind == "huber":
        return torch.nn.functional.huber_loss(err, torch.zeros_like(err), reduction="none", delta=delta)
    if kind == "logcosh":
        z = err / delta
        return delta ** 2 * (z.abs() + torch.nn.functional.softplus(-2 * z.abs()) - math.log(2.0))
    raise ValueError(f"неизвестный loss: {kind}")


DATA_LOSSES = ("mse", "l1", "huber", "logcosh")
GRAD_MODES = ("fixed", "adaptive", "pcgrad")


class Trainer:
    """Обучение ``EOSNet``: loss по данным + λ · физические штрафы.

    λ = 0 для всех штрафов — обычная (vanilla) сеть той же архитектуры:
    честное сравнение «с физикой / без физики».

    Args:
        rho, T, P, e: вся таблица, тензоры (N,) на нужном устройстве.
        idx_train, idx_val: индексы строк (тензоры на том же устройстве).
        w_train: веса точек train (``model.region_weights``), тензор.
    """

    def __init__(self, rho, T, P, e, idx_train, idx_val, w_train):
        self.rho, self.T, self.P, self.e = rho, T, P, e
        self.idx_train, self.idx_val, self.w_train = idx_train, idx_val, w_train

    def data_loss(self, model, idx, w=None, kind="mse"):
        """Взвешенный loss по данным в нормированном пространстве (log10 P, asinh e после RobustScaler)."""
        err = model.forward_norm(self.rho[idx], self.T[idx]) - model.targets(self.P[idx], self.e[idx])
        l = _data_loss_fn(err, kind)
        return (l if w is None else w[:, None] * l).mean()

    @torch.no_grad()
    def val_loss(self, model, chunk=65536):
        """MSE по данным на всём val (без весов) — одна мера для всех вариантов loss."""
        tot = 0.0
        for i in range(0, len(self.idx_val), chunk):
            idx = self.idx_val[i:i + chunk]
            tot += self.data_loss(model, idx).item() * len(idx)
        return tot / len(self.idx_val)

    def physics_loss(self, model, n_coll, generator=None):
        """Штрафы в случайных точках, равномерных по (log ρ, log T).

        Согласованность — средний квадрат невязки. Устойчивость — среднее нарушение
        (линейный штраф): квадрат слишком слабо давит мелкие нарушения знака производной.
        """
        rho_c, T_c = sample_collocation(n_coll, device=self.rho.device, generator=generator)
        r = physics_terms(log_derivatives(model, rho_c, T_c), rho_c, model.norm.E0)
        return {"cons": (r["cons"] ** 2).mean(), "mono_rho": r["mono_rho"].mean(),
                "mono_T": r["mono_T"].mean(), "sound": r["sound"].mean()}

    @staticmethod
    def _flat_grad(loss, params, retain_graph=True):
        g = torch.autograd.grad(loss, params, retain_graph=retain_graph, allow_unused=True)
        return torch.cat([(gi if gi is not None else torch.zeros_like(p)).reshape(-1)
                          for gi, p in zip(g, params)])

    def fit(self, model, lam=None, steps=20_000, batch=8192, n_coll=4096, lr=2e-3,
            loss="mse", grad="fixed", ramp=0.2, adapt_every=100, adapt_alpha=0.1,
            teacher=None, n_teacher=8192, kd_weight=1.0,
            eval_every=500, seed=0, verbose=True):
        """Обучает модель (Adam, косинусное затухание lr), возвращает историю.

        Args:
            model: ``EOSNet`` на устройстве данных.
            lam: веса штрафов {"cons", "mono_rho", "mono_T", "sound"}; None или нули — vanilla.
            steps: число шагов оптимизатора.
            batch: точек данных в шаге.
            n_coll: точек коллокации в шаге (только если есть ненулевые λ).
            lr: начальный learning rate.
            loss: loss по данным — один из ``DATA_LOSSES``.
            grad: как объединять градиенты данных и физики — один из ``GRAD_MODES``:
                fixed — сумма loss с весами λ;
                adaptive — веса подстраиваются так, чтобы норма градиента каждого штрафа
                    была λ × норма градиента данных (Wang, Teng, Perdikaris, 2021);
                pcgrad — если градиенты данных и физики направлены друг против друга,
                    каждый проецируется на плоскость, ортогональную другому (Yu et al., 2020).
            ramp: доля шагов, за которую λ линейно растёт от 0 до заданных значений.
            adapt_every, adapt_alpha: для adaptive — как часто пересчитывать веса и
                коэффициент скользящего среднего.
            teacher: обученная ``EOSNet`` для дистилляции (или None). Тогда к loss добавляется
                kd_weight · MSE между выходами ученика и учителя в n_teacher случайных точках
                всей области — ученик учится и между узлами таблицы.
            eval_every: как часто считать val loss и сохранять лучшее состояние.
            seed: зерно для батчей и коллокации.

        Returns:
            dict: step, train (loss по данным), phys (взвешенный штраф), val, time_s, best_val,
            adapt (последние адаптивные множители). В модель загружается состояние с лучшим val.
        """
        lam = {k: v for k, v in (lam or {}).items() if v}
        dev = self.rho.device
        gen = torch.Generator(device=dev).manual_seed(seed)
        params = [p for p in model.parameters() if p.requires_grad]
        opt = torch.optim.Adam(params, lr=lr)
        sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=steps, eta_min=lr * 1e-3)
        hist = {"step": [], "train": [], "phys": [], "val": []}
        adapt = {k: 1.0 for k in lam}
        best, best_state, t0 = float("inf"), None, time.time()
        zero = torch.zeros((), device=dev)
        run_d, run_p, run_n = zero.clone(), zero.clone(), 0
        for step in range(1, steps + 1):
            b = torch.randint(0, len(self.idx_train), (batch,), device=dev, generator=gen)
            loss_d = self.data_loss(model, self.idx_train[b], self.w_train[b], kind=loss)
            if teacher is not None:
                rho_k, T_k = sample_collocation(n_teacher, device=dev, generator=gen)
                with torch.no_grad():
                    y_t = teacher.forward_norm(rho_k, T_k)
                loss_d = loss_d + kd_weight * ((model.forward_norm(rho_k, T_k) - y_t) ** 2).mean()
            loss_p = torch.zeros((), device=dev)
            opt.zero_grad(set_to_none=True)
            if not lam:
                loss_d.backward()
            else:
                terms = self.physics_loss(model, n_coll, generator=gen)
                warm = min(1.0, step / max(1, ramp * steps))
                if grad == "adaptive" and (step == 1 or step % adapt_every == 0):
                    g_d = self._flat_grad(loss_d, params).norm()
                    for k in lam:
                        g_k = self._flat_grad(terms[k], params).norm()
                        ratio = float(torch.clamp(g_d / (g_k + 1e-12), max=1e2))
                        adapt[k] = (1 - adapt_alpha) * adapt[k] + adapt_alpha * ratio
                mult = adapt if grad == "adaptive" else {k: 1.0 for k in lam}
                loss_p = warm * sum(lam[k] * mult[k] * terms[k] for k in lam)
                if grad == "pcgrad":
                    g_d = self._flat_grad(loss_d, params)
                    g_p = self._flat_grad(loss_p, params, retain_graph=False)
                    dot = torch.clamp(torch.dot(g_d, g_p), max=0.0)   # 0, если конфликта нет
                    g_d, g_p = (g_d - dot / (g_p.dot(g_p) + 1e-20) * g_p,
                                g_p - dot / (g_d.dot(g_d) + 1e-20) * g_d)
                    g, i = g_d + g_p, 0
                    for p in params:
                        p.grad = g[i:i + p.numel()].view_as(p).clone(); i += p.numel()
                elif grad in ("fixed", "adaptive"):
                    (loss_d + loss_p).backward()
                else:
                    raise ValueError(f"неизвестный grad: {grad}")
            for p in params:   # NaN/∞ в градиенте одного шага не должны испортить веса
                if p.grad is not None:
                    torch.nan_to_num_(p.grad, nan=0.0, posinf=0.0, neginf=0.0)
            opt.step()
            sched.step()
            run_d += loss_d.detach(); run_p += loss_p.detach(); run_n += 1
            if step % eval_every == 0 or step == steps:
                va = self.val_loss(model)
                hist["step"].append(step); hist["val"].append(va)
                hist["train"].append(run_d.item() / run_n); hist["phys"].append(run_p.item() / run_n)
                run_d, run_p, run_n = zero.clone(), zero.clone(), 0
                if va < best:
                    best, best_state = va, copy.deepcopy(model.state_dict())
                if verbose and (step % (eval_every * 2) == 0 or step == steps):
                    print(f"  шаг {step:6d}  train {hist['train'][-1]:.3e}  phys {hist['phys'][-1]:.3e}  "
                          f"val {va:.3e}  {time.time() - t0:.0f} с")
        if best_state is None:
            raise RuntimeError("val loss всё время NaN/∞ — обучение разошлось; уменьшите lr "
                               "(например, lr=5e-4) или веса штрафов")
        model.load_state_dict(best_state)
        hist["time_s"] = time.time() - t0
        hist["best_val"] = best
        hist["adapt"] = adapt
        return hist


def numpy_predict(model: EOSNet, chunk: int = 262_144):
    """Функция (rho, T) → (P, e) на numpy для ``metrics.evaluate``.

    Считает в float64 на копии модели: в метриках нарушения физики ищутся конечными
    разностями с шагом 1e-4, и шума float32 там было бы достаточно для ложных нарушений.
    """
    m64 = copy.deepcopy(model).double().eval()
    dev = next(m64.parameters()).device

    @torch.no_grad()
    def predict(rho, T):
        rho = torch.tensor(np.asarray(rho, dtype=np.float64), device=dev)
        T = torch.tensor(np.asarray(T, dtype=np.float64), device=dev)
        P, e = [], []
        for i in range(0, len(rho), chunk):
            p_, e_ = m64(rho[i:i + chunk], T[i:i + chunk])
            P.append(p_.cpu().numpy()); e.append(e_.cpu().numpy())
        return np.concatenate(P), np.concatenate(e)

    return predict


def physics_report(model: EOSNet, n: int = 100_000, seed: int = 1, chunk: int = 32_768) -> dict[str, float]:
    """Нарушения физики через autograd (float64) в n случайных точках домена.

    Returns:
        Доли точек в процентах: dP/dρ ≤ 0, de/dT < 0, c_s² ≤ 0, c_s ≥ c,
        |невязка согласованности| > 0.1; и медиана |невязки согласованности|.
    """
    m64 = copy.deepcopy(model).double().eval()
    dev = next(m64.parameters()).device
    gen = torch.Generator(device=dev).manual_seed(seed)
    rho, T = sample_collocation(n, device=dev, generator=gen)
    rho, T = rho.double(), T.double()
    out = {k: [] for k in ["cons", "a", "e_t", "cs2"]}
    for i in range(0, n, chunk):
        r_, T_ = rho[i:i + chunk], T[i:i + chunk]
        d = log_derivatives(m64, r_, T_, create_graph=False)
        out["cons"].append(physics_terms(d, r_, m64.norm.E0)["cons"].detach())
        out["a"].append(d["a"].detach()); out["e_t"].append(d["e_t"].detach())
        out["cs2"].append(sound_speed2(d, r_).detach())
    o = {k: torch.cat(v).cpu().numpy() for k, v in out.items()}
    return {
        "viol_dP_drho_%": 100 * float(np.mean(o["a"] <= 0)),
        "viol_de_dT_%": 100 * float(np.mean(o["e_t"] < 0)),
        "viol_cs2_%": 100 * float(np.mean(~(o["cs2"] > 0))),
        "viol_superluminal_%": 100 * float(np.mean(o["cs2"] >= C_LIGHT_KMS ** 2)),
        "viol_cons_gt_0.1_%": 100 * float(np.mean(np.abs(o["cons"]) > 0.1)),
        "cons_median": float(np.median(np.abs(o["cons"]))),
    }


def table_consistency(df) -> "pd.DataFrame":
    """Невязка согласованности в самой таблице (конечные разности по узлам сетки).

    Та же величина, что ``physics_terms(...)["cons"]``: (A − (1 − g)) / (|A| + 1 + |g|), A = ρ ∂e/∂lnρ / P.

    Args:
        df: таблица из ``load_all()``.

    Returns:
        DataFrame по областям: медиана и 95-й перцентиль |невязки|, доля |невязки| > 0.1,
        медиана отношения ρ²∂e/∂ρ / (P − T∂P/∂T) (1 — согласовано).
    """
    import pandas as pd
    from eos.data import region_grid

    rows = []
    for name, g_df in df.groupby("region"):
        rho, T, P, e = region_grid(g_df)
        lr, lt = np.log(rho), np.log(T)
        e_r = np.gradient(e, lr, axis=1)[1:-1, 1:-1]
        g = np.gradient(np.log(P), lt, axis=0)[1:-1, 1:-1]
        Pi, r = P[1:-1, 1:-1], rho[None, 1:-1]
        A = r * e_r / Pi
        res = np.abs((A - 1 + g) / (np.abs(A) + 1 + np.abs(g)))
        with np.errstate(divide="ignore", invalid="ignore"):
            ratio = r * e_r / (Pi * (1 - g))
        rows.append({"region": name, "cons_median": np.median(res), "cons_p95": np.percentile(res, 95),
                     "frac_gt_0.1_%": 100 * np.mean(res > 0.1), "ratio_median": np.nanmedian(ratio)})
    return pd.DataFrame(rows)


STAB_COLS = ("viol_dP_drho_%", "viol_de_dT_%", "viol_cs2_%")


def run_variant(trainer: Trainer, norm, cfg: dict, val_df, steps: int, device,
                n_phys: int = 50_000, seed: int = 0, verbose: bool = False):
    """Обучает один вариант и оценивает его на val.

    Args:
        trainer: ``Trainer`` с данными.
        norm: ``EOSNormalizer``.
        cfg: hidden, layers, activation, loss, grad, lam (веса штрафов);
            необязательно lr, batch, n_coll.
        val_df: строки val для ``metrics.evaluate``.
        steps: число шагов обучения.
        device: устройство torch.
        n_phys: точек для ``physics_report``.
        seed: зерно (инициализация, батчи, коллокация).

    Returns:
        (model, hist, row): модель, история обучения, строка с метриками на val.
    """
    from eos.metrics import evaluate

    torch.manual_seed(seed)
    m = EOSNet(norm, hidden=cfg["hidden"], layers=cfg["layers"], activation=cfg["activation"]).to(device)
    h = trainer.fit(m, lam=cfg["lam"], steps=steps, loss=cfg["loss"], grad=cfg["grad"],
                    lr=cfg.get("lr", 2e-3), batch=cfg.get("batch", 8192), n_coll=cfg.get("n_coll", 4096),
                    seed=seed, verbose=verbose)
    met, _ = evaluate(numpy_predict(m), val_df, "")
    row = {"time_s": h["time_s"], "val_mse": h["best_val"],
           **{k: met[k] for k in met if k.startswith("err_")},
           **physics_report(m, n=n_phys)}
    return m, h, row


def score(df) -> "pd.Series":
    """Чем меньше, тем лучше: медианы ошибок P и e плюс 0.1 × их 99-е перцентили (%)."""
    return (df["err_P_median_%"] + df["err_e_median_%"]
            + 0.1 * (df["err_P_p99_%"] + df["err_e_p99_%"]))


def pick_best(df, max_viol: float = 0.1) -> str:
    """Лучший вариант: минимальный ``score`` среди тех, где устойчивость (∂P/∂ρ > 0,
    ∂e/∂T ≥ 0, c_s² > 0) нарушена меньше чем в max_viol % точек; если таких нет —
    вариант с наименьшими нарушениями.
    """
    stab = df[list(STAB_COLS)].max(axis=1)
    ok = stab < max_viol
    return score(df[ok]).idxmin() if ok.any() else stab.idxmin()


def search(trainer: Trainer, norm, base: dict, variants: dict, val_df, steps: int, device,
           n_phys: int = 50_000, seed: int = 0, verbose: bool = True):
    """Перебор: каждый вариант — ``base``, в котором заменены указанные поля.

    Args:
        base: конфигурация по умолчанию (см. ``run_variant``).
        variants: {имя: {поле: значение}}.

    Returns:
        (table, models, hists): DataFrame по вариантам (+ колонка score),
        словари моделей и историй обучения.
    """
    import pandas as pd

    rows, models, hists = [], {}, {}
    for name, over in variants.items():
        cfg = {**base, **over}
        print(f"=== {name}: " + ", ".join(f"{k}={cfg[k]}" for k in over))
        models[name], hists[name], row = run_variant(trainer, norm, cfg, val_df, steps, device,
                                                     n_phys=n_phys, seed=seed, verbose=verbose)
        models[name] = models[name].cpu()  # освобождаем память GPU
        rows.append({"variant": name, **{k: str(v) for k, v in over.items()}, **row})
        print("    P med {err_P_median_%:.3f}%  e med {err_e_median_%:.3f}%  "
              "∂P/∂ρ≤0 {viol_dP_drho_%:.3f}%  ∂e/∂T<0 {viol_de_dT_%:.3f}%  {time_s:.0f} с".format(**row))
    table = pd.DataFrame(rows).set_index("variant")
    table["score"] = score(table)
    return table, models, hists
