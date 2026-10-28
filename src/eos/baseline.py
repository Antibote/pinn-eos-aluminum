"""Бейзлайн: билинейная интерполяция табличного УРС по областям.

Каждая из 20 областей — своя прямоугольная сетка, поэтому интерполятор строится
для каждой области отдельно. Интерполяция идёт в координатах (log10 T, log10 rho):
давление интерполируется как log10 P (оно всегда > 0 и меняется на 12 порядков),
энергия E — как есть (бывает отрицательной).

Диапазоны областей полуоткрытые: последний узел области лежит чуть ниже границы
с соседней (например, 0.0971 при границе 0.1). Запросы в этом зазоре
обслуживаются линейной экстраполяцией с краёв сетки области.
"""

from __future__ import annotations

import time

import numpy as np
import pandas as pd
from scipy.interpolate import RegularGridInterpolator

from eos.data import region_grid

RHO_EDGES = np.array([1e-3, 1e-1, 1.5, 5.0, 1e4])
T_EDGES = np.array([0.1, 1e2, 3e3, 3e4, 1e5, 1e7])
RHO_BANDS = ["A", "B", "C", "D"]


def region_name(i_rho: int, i_T: int) -> str:
    """Имя файла области по номеру диапазона плотности и температуры.

    Args:
        i_rho: номер диапазона плотности, 0..3 (rho_A..rho_D).
        i_T: номер диапазона температуры, 0..4 (T_1..T_5).

    Returns:
        Имя области, например "Obl07" для (1, 1).
    """
    return f"Obl{i_rho * 5 + i_T + 1:02d}"


def decimate_mask(n_T: int, n_rho: int) -> np.ndarray:
    """Маска узлов для проверки интерполяции: каждый второй узел уходит в тест.

    В обучающую сетку попадают чётные узлы по обеим осям (и всегда последний,
    чтобы сетка покрывала всю область), остальные узлы — тестовые.

    Args:
        n_T: число узлов по T.
        n_rho: число узлов по rho.

    Returns:
        Булев массив формы (n_T, n_rho): True — узел тестовый.
    """
    keep_T = np.unique(np.r_[np.arange(0, n_T, 2), n_T - 1])
    keep_rho = np.unique(np.r_[np.arange(0, n_rho, 2), n_rho - 1])
    test = np.ones((n_T, n_rho), dtype=bool)
    test[np.ix_(keep_T, keep_rho)] = False
    return test


class BilinearEOS:
    """Билинейная интерполяция P(rho, T) и E(rho, T) по 20 областям таблицы.

    Args:
        df: таблица из ``load_all()`` с колонками rho, T, P, E, region.
        decimate: если True, интерполяторы строятся только по обучающим узлам
            ``decimate_mask`` — для честной оценки точности на тестовых узлах.
    """

    def __init__(self, df: pd.DataFrame, decimate: bool = False):
        self.interp = {}
        for name, g in df.groupby("region"):
            rho, T, P, E = region_grid(g)
            if decimate:
                test = decimate_mask(len(T), len(rho))
                keep_T = ~test.all(axis=1)
                keep_rho = ~test.all(axis=0)
                rho, T = rho[keep_rho], T[keep_T]
                P, E = P[np.ix_(keep_T, keep_rho)], E[np.ix_(keep_T, keep_rho)]
            axes = (np.log10(T), np.log10(rho))
            self.interp[name] = (
                RegularGridInterpolator(axes, np.log10(P), bounds_error=False, fill_value=None),
                RegularGridInterpolator(axes, E, bounds_error=False, fill_value=None),
            )

    def predict(self, rho: np.ndarray, T: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        """Предсказание давления и энергии.

        Args:
            rho: плотность, г/см³, форма (N,).
            T: температура, K, форма (N,).

        Returns:
            (P, E): давление (ГПа) и удельная внутренняя энергия (кДж/г), форма (N,).

        Raises:
            ValueError: если rho или T <= 0.
        """
        rho = np.asarray(rho, dtype=float)
        T = np.asarray(T, dtype=float)
        if (rho <= 0).any() or (T <= 0).any():
            raise ValueError("rho и T должны быть > 0")
        i_rho = np.clip(np.searchsorted(RHO_EDGES, rho, side="right") - 1, 0, 3)
        i_T = np.clip(np.searchsorted(T_EDGES, T, side="right") - 1, 0, 4)
        q = np.column_stack([np.log10(T), np.log10(rho)])
        P = np.empty_like(rho)
        E = np.empty_like(rho)
        region_id = i_rho * 5 + i_T
        for rid in np.unique(region_id):
            m = region_id == rid
            f_P, f_E = self.interp[region_name(rid // 5, rid % 5)]
            P[m] = 10 ** f_P(q[m])
            E[m] = f_E(q[m])
        return P, E


def holdout_errors(df: pd.DataFrame) -> pd.DataFrame:
    """Ошибки интерполяции на узлах, не вошедших в прореженную сетку.

    Args:
        df: таблица из ``load_all()``.

    Returns:
        DataFrame тестовых узлов: rho, T, P, E, region, P_pred, E_pred,
        err_P — относительная ошибка |ΔP|/P,
        err_E — ошибка |ΔE| / max(|E|, 1 кДж/г) (относительная, но без деления
        на почти ноль там, где энергия меняет знак).
    """
    model = BilinearEOS(df, decimate=True)
    parts = []
    for name, g in df.groupby("region"):
        rho, T, _, _ = region_grid(g)
        test = decimate_mask(len(T), len(rho))
        TT, RR = np.meshgrid(T, rho, indexing="ij")
        q = pd.DataFrame({"T": TT[test], "rho": RR[test]})
        parts.append(q.merge(g, on=["rho", "T"]))
    t = pd.concat(parts, ignore_index=True)
    t["P_pred"], t["E_pred"] = model.predict(t["rho"].to_numpy(), t["T"].to_numpy())
    t["err_P"] = (t["P_pred"] - t["P"]).abs() / t["P"].abs()
    t["err_E"] = (t["E_pred"] - t["E"]).abs() / np.maximum(t["E"].abs(), 1.0)
    return t


def summarize_errors(t: pd.DataFrame) -> pd.DataFrame:
    """Сводка ошибок: медиана, среднее, 99-й перцентиль и максимум, в процентах.

    Args:
        t: результат ``holdout_errors``.

    Returns:
        DataFrame со строками err_P и err_E.
    """
    s = t[["err_P", "err_E"]].agg(["median", "mean", lambda x: x.quantile(0.99), "max"]).T * 100
    s.columns = ["median, %", "mean, %", "p99, %", "max, %"]
    return s


def random_queries(n: int = 100_000, seed: int = 0) -> tuple[np.ndarray, np.ndarray]:
    """Случайные запросы, равномерные по log rho и log T во всей таблице.

    Args:
        n: число запросов.
        seed: зерно генератора.

    Returns:
        (rho, T), форма (n,).
    """
    rng = np.random.default_rng(seed)
    rho = 10 ** rng.uniform(np.log10(RHO_EDGES[0]), np.log10(RHO_EDGES[-1]), n)
    T = 10 ** rng.uniform(np.log10(T_EDGES[0]), np.log10(T_EDGES[-1]), n)
    return rho, T


def benchmark(model, n: int = 100_000, repeats: int = 5) -> float:
    """Время ответа модели на n случайных запросов (лучшее из repeats), в секундах.

    Args:
        model: объект с методом ``predict(rho, T)``.
        n: число запросов.
        repeats: число повторов.

    Returns:
        Время одного прогона n запросов, с.
    """
    rho, T = random_queries(n)
    best = np.inf
    for _ in range(repeats):
        t0 = time.perf_counter()
        model.predict(rho, T)
        best = min(best, time.perf_counter() - t0)
    return best
