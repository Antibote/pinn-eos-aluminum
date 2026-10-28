"""Единые метрики для сравнения baseline / Vanilla / PINN.

Все модели оцениваются одной функцией ``evaluate`` на одних и тех же точках
(``split["test"]`` из раздела 1.3), поэтому цифры в итоговой таблице сравнимы.
"""

from __future__ import annotations

from typing import Callable

import numpy as np
import pandas as pd

C_LIGHT_KMS = 299_792.458  # скорость света, км/с; c_s² в кДж/г = (км/с)²

Predict = Callable[[np.ndarray, np.ndarray], tuple[np.ndarray, np.ndarray]]


def relative_errors(P_pred: np.ndarray, e_pred: np.ndarray,
                    P: np.ndarray, e: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Ошибки предсказания.

    Args:
        P_pred, e_pred: предсказанные давление (ГПа) и энергия (кДж/г).
        P, e: эталонные значения из таблицы.

    Returns:
        (err_P, err_e): |ΔP| / P и |Δe| / max(|e|, 1 кДж/г). У энергии знаменатель
        ограничен снизу, потому что e меняет знак и обычная относительная ошибка
        около нуля уходит в бесконечность.
    """
    err_P = np.abs(P_pred - P) / np.abs(P)
    err_e = np.abs(e_pred - e) / np.maximum(np.abs(e), 1.0)
    return err_P, err_e


def physics_violations(predict: Predict, rho: np.ndarray, T: np.ndarray,
                       h: float = 1e-4) -> dict[str, float]:
    """Доля точек, где модель нарушает термодинамические ограничения.

    Производные считаются центральными разностями с относительным шагом h,
    одинаково для любой модели (таблица, интерполяция, нейросеть).

    Args:
        predict: функция (rho, T) → (P, e).
        rho: плотность, г/см³, форма (N,).
        T: температура, K, форма (N,).
        h: относительный шаг разностей.

    Returns:
        Доли в процентах: viol_dP_drho_% — (∂P/∂ρ)_T ≤ 0,
        viol_de_dT_% — (∂e/∂T)_ρ < 0, viol_dP_dT_% — (∂P/∂T)_ρ < 0,
        viol_cs_% — не выполнено 0 < c_s < c, c_s² = ∂P/∂ρ + T (∂P/∂T)² / (ρ² ∂e/∂T).
    """
    P_rp, _ = predict(rho * (1 + h), T)
    P_rm, _ = predict(rho * (1 - h), T)
    P_tp, e_tp = predict(rho, T * (1 + h))
    P_tm, e_tm = predict(rho, T * (1 - h))
    dP_drho = (P_rp - P_rm) / (2 * h * rho)
    dP_dT = (P_tp - P_tm) / (2 * h * T)
    de_dT = (e_tp - e_tm) / (2 * h * T)
    with np.errstate(divide="ignore", invalid="ignore"):
        cs2 = dP_drho + T * dP_dT ** 2 / (rho ** 2 * de_dT)
    return {
        "viol_cs_%": 100 * float(np.mean(~((cs2 > 0) & (cs2 < C_LIGHT_KMS ** 2)))),
        "viol_dP_drho_%": 100 * float(np.mean(dP_drho <= 0)),
        "viol_de_dT_%": 100 * float(np.mean(de_dT < 0)),
        "viol_dP_dT_%": 100 * float(np.mean(dP_dT < 0)),
    }


def evaluate(predict: Predict, df_test: pd.DataFrame, name: str) -> tuple[dict, pd.DataFrame]:
    """Точность и нарушения физики модели на тестовых точках.

    Args:
        predict: функция (rho, T) → (P, e) в физических единицах.
        df_test: тестовые строки таблицы (колонки rho, T, P, e, region).
        name: имя модели для итоговой таблицы.

    Returns:
        (metrics, per_point): словарь метрик (ошибки в процентах) и таблица
        тестовых точек с колонками P_pred, e_pred, err_P, err_e.
    """
    rho = df_test["rho"].to_numpy()
    T = df_test["T"].to_numpy()
    P_pred, e_pred = predict(rho, T)
    err_P, err_e = relative_errors(P_pred, e_pred, df_test["P"].to_numpy(), df_test["e"].to_numpy())
    metrics = {"model": name, "n_test": len(df_test)}
    for key, err in [("P", err_P), ("e", err_e)]:
        metrics[f"err_{key}_median_%"] = 100 * float(np.median(err))
        metrics[f"err_{key}_mean_%"] = 100 * float(np.mean(err))
        metrics[f"err_{key}_p99_%"] = 100 * float(np.quantile(err, 0.99))
        metrics[f"err_{key}_max_%"] = 100 * float(np.max(err))
    metrics.update(physics_violations(predict, rho, T))
    per_point = df_test.assign(P_pred=P_pred, e_pred=e_pred, err_P=err_P, err_e=err_e)
    return metrics, per_point
