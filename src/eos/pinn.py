"""PINN: физические штрафы, обучение и проверка физики для ``EOSNet``.

Производные берутся по логарифмам: ln ρ и ln T. Так все штрафы безразмерны и
одного порядка во всём диапазоне (ρ — 7 порядков, T — 8, P — 12).

Ограничения (их ``physics_terms`` возвращает как невязки, 0 — выполнено):
- согласованность (∂e/∂ρ)_T = (P − T(∂P/∂T)_ρ) / ρ²; в лог-переменных
  ρ ∂e/∂lnρ = P (1 − g), g = ∂lnP/∂lnT; невязка нормирована на P (1 + |g|);
- механическая устойчивость (∂P/∂ρ)_T > 0  ⇔  ∂lnP/∂lnρ > 0;
- термическая устойчивость (∂e/∂T)_ρ ≥ 0 (нестрого: в таблице есть плато при T ≲ 7 K);
- скорость звука c_s² = ∂P/∂ρ + T(∂P/∂T)² / (ρ² ∂e/∂T), 0 < c_s < c — проверяется
  в ``physics_report`` (в loss не входит: следует из двух предыдущих при c_s ≪ c).

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

LN10 = math.log(10.0)
C_LIGHT_KMS = 299_792.458  # c, км/с; c_s² в кДж/г = (км/с)²


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
        cons — относительная невязка согласованности;
        mono_rho — max(0, −∂lnP/∂lnρ); mono_T — max(0, −∂e/∂lnT) / (|e| + E0).
    """
    P, g = d["P"], d["g"]
    cons = (rho * d["e_r"] / P - 1 + g) / (1 + g.abs())
    mono_rho = torch.relu(-d["a"])
    mono_T = torch.relu(-d["e_t"]) / (d["e"].abs() + E0)
    return {"cons": cons, "mono_rho": mono_rho, "mono_T": mono_T}


def sound_speed2(d: dict[str, torch.Tensor], rho: torch.Tensor) -> torch.Tensor:
    """c_s² [кДж/г = (км/с)²] = P a / ρ + (P g)² / (ρ² ∂e/∂lnT)."""
    P = d["P"]
    return P * d["a"] / rho + (P * d["g"]) ** 2 / (rho ** 2 * d["e_t"])


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

    def data_loss(self, model, idx, w=None):
        """Взвешенная MSE в нормированном пространстве (log10 P, asinh e после RobustScaler)."""
        err = (model.forward_norm(self.rho[idx], self.T[idx]) - model.targets(self.P[idx], self.e[idx])) ** 2
        return (err if w is None else w[:, None] * err).mean()

    @torch.no_grad()
    def val_loss(self, model, chunk=65536):
        """Loss по данным на всём val (без весов)."""
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
        return {"cons": (r["cons"] ** 2).mean(), "mono_rho": r["mono_rho"].mean(), "mono_T": r["mono_T"].mean()}

    def fit(self, model, lam=None, steps=20_000, batch=8192, n_coll=4096, lr=2e-3,
            ramp=0.2, eval_every=500, seed=0, verbose=True):
        """Обучает модель (Adam, косинусное затухание lr), возвращает историю.

        Args:
            model: ``EOSNet`` на устройстве данных.
            lam: веса штрафов {"cons", "mono_rho", "mono_T"}; None или нули — vanilla.
            steps: число шагов оптимизатора.
            batch: точек данных в шаге.
            n_coll: точек коллокации в шаге (только если есть ненулевые λ).
            lr: начальный learning rate.
            ramp: доля шагов, за которую λ линейно растёт от 0 до заданных значений
                (сначала сеть подгоняется под данные, потом подключается физика).
            eval_every: как часто считать val loss и сохранять лучшее состояние.
            seed: зерно для батчей и коллокации.

        Returns:
            dict: step, train (loss по данным), phys (взвешенный штраф), val, time_s, best_val.
            В модель загружается состояние с лучшим val loss.
        """
        lam = {k: v for k, v in (lam or {}).items() if v}
        dev = self.rho.device
        gen = torch.Generator(device=dev).manual_seed(seed)
        opt = torch.optim.Adam(model.parameters(), lr=lr)
        sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=steps, eta_min=lr * 1e-3)
        hist = {"step": [], "train": [], "phys": [], "val": []}
        best, best_state, t0 = float("inf"), None, time.time()
        run_d, run_p, run_n = 0.0, 0.0, 0
        for step in range(1, steps + 1):
            b = torch.randint(0, len(self.idx_train), (batch,), device=dev, generator=gen)
            loss_d = self.data_loss(model, self.idx_train[b], self.w_train[b])
            loss_p = torch.zeros((), device=dev)
            if lam:
                terms = self.physics_loss(model, n_coll, generator=gen)
                warm = min(1.0, step / max(1, ramp * steps))
                loss_p = warm * sum(lam[k] * terms[k] for k in lam)
            opt.zero_grad(set_to_none=True)
            (loss_d + loss_p).backward()
            opt.step()
            sched.step()
            run_d += loss_d.item(); run_p += loss_p.item(); run_n += 1
            if step % eval_every == 0 or step == steps:
                va = self.val_loss(model)
                hist["step"].append(step); hist["val"].append(va)
                hist["train"].append(run_d / run_n); hist["phys"].append(run_p / run_n)
                run_d, run_p, run_n = 0.0, 0.0, 0
                if va < best:
                    best, best_state = va, copy.deepcopy(model.state_dict())
                if verbose and (step % (eval_every * 10) == 0 or step == steps):
                    print(f"  шаг {step:6d}  train {hist['train'][-1]:.3e}  phys {hist['phys'][-1]:.3e}  "
                          f"val {va:.3e}  {time.time() - t0:.0f} с")
        model.load_state_dict(best_state)
        hist["time_s"] = time.time() - t0
        hist["best_val"] = best
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

    Та же величина, что ``physics_terms(...)["cons"]``: (ρ ∂e/∂lnρ − P(1 − g)) / (P (1 + |g|)).

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
        res = np.abs((r * e_r / Pi - 1 + g) / (1 + np.abs(g)))
        with np.errstate(divide="ignore", invalid="ignore"):
            ratio = r * e_r / (Pi * (1 - g))
        rows.append({"region": name, "cons_median": np.median(res), "cons_p95": np.percentile(res, 95),
                     "frac_gt_0.1_%": 100 * np.mean(res > 0.1), "ratio_median": np.nanmedian(ratio)})
    return pd.DataFrame(rows)
