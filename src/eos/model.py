"""Нормировка, разбиение данных и нейросетевая модель УРС.

Преобразования (выводы EDA, раздел 1.1):
- входы: log10 rho и log10 T, линейно в [-1, 1] по физическим границам таблицы;
- давление: log10 P, затем z-score;
- энергия: asinh(e / e_scale) — «логарифм со знаком», затем z-score.

Все преобразования выполняются внутри ``EOSNet``: на вход подаются физические
rho [г/см³] и T [K], на выходе — P [ГПа] и e [кДж/г]. Поэтому autograd сразу
даёт производные в физических единицах (``derivatives``).
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
import torch

from eos.baseline import decimate_mask
from eos.data import region_grid

LOG_RHO_BOUNDS = (-3.0, 4.0)  # log10 rho: 1e-3 … 1e4 г/см³
LOG_T_BOUNDS = (-1.0, 7.0)  # log10 T: 0.1 … 1e7 K


def make_split(df: pd.DataFrame, val_frac: float = 0.2, seed: int = 0) -> dict[str, np.ndarray]:
    """Разбиение на train / val / test, согласованное с бейзлайном.

    train — узлы прореженной сетки (каждый второй по rho и T, как в
    ``baseline.decimate_mask``), остальные узлы случайно делятся на val и test.
    Так ошибка сети на test сравнима с ошибкой билинейной интерполяции.

    Args:
        df: таблица из ``load_all()``.
        val_frac: доля отложенных узлов, которая идёт в val.
        seed: зерно генератора.

    Returns:
        Словарь {"train", "val", "test"} с индексами строк df.
    """
    is_test = np.zeros(len(df), dtype=bool)
    for _, g in df.groupby("region"):
        rho, T, _, _ = region_grid(g)
        mask = decimate_mask(len(T), len(rho))
        i_T = np.searchsorted(T, g["T"].to_numpy())
        i_rho = np.searchsorted(rho, g["rho"].to_numpy())
        is_test[df.index.get_indexer(g.index)] = mask[i_T, i_rho]
    held = np.flatnonzero(is_test)
    rng = np.random.default_rng(seed)
    rng.shuffle(held)
    n_val = int(len(held) * val_frac)
    return {
        "train": np.flatnonzero(~is_test),
        "val": np.sort(held[:n_val]),
        "test": np.sort(held[n_val:]),
    }


def region_weights(df: pd.DataFrame) -> np.ndarray:
    """Веса точек, выравнивающие плотность сетки по областям.

    Вес точки = площадь её области в координатах (log10 rho, log10 T) / число
    точек области в df. Нормированы на среднее 1. Без весов полосы с линейной
    сеткой (rho 1.5–5, T 3e4–1e5: ~7% диапазона, ~30% точек) доминируют в loss.

    Args:
        df: строки таблицы (например, только train).

    Returns:
        Массив весов формы (len(df),).
    """
    g = df.groupby("region")
    area = (np.log10(g["rho"].max()) - np.log10(g["rho"].min())) * \
           (np.log10(g["T"].max()) - np.log10(g["T"].min()))
    w = (area / g.size()).reindex(df["region"]).to_numpy()
    return w / w.mean()


def compute_normalization(df: pd.DataFrame, weights: np.ndarray, e_scale: float = 1.0) -> dict:
    """Коэффициенты нормировки (взвешенные среднее и std выходов).

    Args:
        df: обучающие строки.
        weights: веса из ``region_weights``.
        e_scale: масштаб энергии в asinh(e / e_scale), кДж/г.

    Returns:
        Словарь для ``normalization.json``.
    """
    lP = np.log10(df["P"].to_numpy())
    ae = np.arcsinh(df["e"].to_numpy() / e_scale)

    def wstat(x):
        m = np.average(x, weights=weights)
        return float(m), float(np.sqrt(np.average((x - m) ** 2, weights=weights)))

    lP_mean, lP_std = wstat(lP)
    ae_mean, ae_std = wstat(ae)
    return {
        "lrho_min": LOG_RHO_BOUNDS[0], "lrho_max": LOG_RHO_BOUNDS[1],
        "lT_min": LOG_T_BOUNDS[0], "lT_max": LOG_T_BOUNDS[1],
        "lP_mean": lP_mean, "lP_std": lP_std,
        "e_scale": e_scale, "ae_mean": ae_mean, "ae_std": ae_std,
        "units": {"rho": "g/cm^3", "T": "K", "P": "GPa", "e": "kJ/g"},
        "transforms": {
            "x_rho": "2*(log10(rho)-lrho_min)/(lrho_max-lrho_min)-1",
            "x_T": "2*(log10(T)-lT_min)/(lT_max-lT_min)-1",
            "y_P": "(log10(P)-lP_mean)/lP_std",
            "y_e": "(asinh(e/e_scale)-ae_mean)/ae_std",
        },
    }


def save_normalization(norm: dict, path: str | Path) -> None:
    """Сохраняет коэффициенты нормировки в JSON."""
    with open(path, "w") as f:
        json.dump(norm, f, indent=2, ensure_ascii=False)


def load_normalization(path: str | Path) -> dict:
    """Читает коэффициенты нормировки из JSON."""
    with open(path) as f:
        return json.load(f)


class EOSNet(torch.nn.Module):
    """MLP (rho, T) → (P, e) с нормировкой внутри.

    Args:
        norm: словарь из ``compute_normalization`` / ``normalization.json``.
        hidden: ширина скрытых слоёв.
        layers: число скрытых слоёв.
    """

    def __init__(self, norm: dict, hidden: int = 128, layers: int = 4):
        super().__init__()
        self.norm = {k: v for k, v in norm.items() if isinstance(v, (int, float))}
        dims = [2] + [hidden] * layers + [2]
        mods = []
        for a, b in zip(dims[:-1], dims[1:]):
            mods += [torch.nn.Linear(a, b), torch.nn.SiLU()]
        self.net = torch.nn.Sequential(*mods[:-1])  # без активации на выходе

    def inputs(self, rho: torch.Tensor, T: torch.Tensor) -> torch.Tensor:
        """Физические rho, T → нормированные входы в [-1, 1], форма (N, 2)."""
        n = self.norm
        x_rho = 2 * (torch.log10(rho) - n["lrho_min"]) / (n["lrho_max"] - n["lrho_min"]) - 1
        x_T = 2 * (torch.log10(T) - n["lT_min"]) / (n["lT_max"] - n["lT_min"]) - 1
        return torch.stack([x_rho, x_T], dim=-1)

    def targets(self, P: torch.Tensor, e: torch.Tensor) -> torch.Tensor:
        """Физические P, e → нормированные выходы, форма (N, 2). Для loss по данным."""
        n = self.norm
        y_P = (torch.log10(P) - n["lP_mean"]) / n["lP_std"]
        y_e = (torch.asinh(e / n["e_scale"]) - n["ae_mean"]) / n["ae_std"]
        return torch.stack([y_P, y_e], dim=-1)

    def forward_norm(self, rho: torch.Tensor, T: torch.Tensor) -> torch.Tensor:
        """Нормированные выходы сети, форма (N, 2)."""
        return self.net(self.inputs(rho, T))

    def forward(self, rho: torch.Tensor, T: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """Предсказание в физических единицах.

        Args:
            rho: плотность, г/см³, форма (N,).
            T: температура, K, форма (N,).

        Returns:
            (P, e): давление (ГПа) и энергия (кДж/г), форма (N,).
        """
        n = self.norm
        y = self.forward_norm(rho, T)
        P = 10 ** (y[:, 0] * n["lP_std"] + n["lP_mean"])
        e = n["e_scale"] * torch.sinh(y[:, 1] * n["ae_std"] + n["ae_mean"])
        return P, e


def derivatives(model: EOSNet, rho: torch.Tensor, T: torch.Tensor,
                create_graph: bool = True) -> dict[str, torch.Tensor]:
    """P, e и их частные производные по rho и T через autograd.

    Args:
        model: модель ``EOSNet``.
        rho: плотность, г/см³, форма (N,).
        T: температура, K, форма (N,).
        create_graph: True — производные можно использовать в loss (PINN).

    Returns:
        Словарь с ключами P, e, dP_drho, dP_dT, de_drho, de_dT (физические единицы).
    """
    rho = rho.detach().requires_grad_(True)
    T = T.detach().requires_grad_(True)
    P, e = model(rho, T)
    dP_drho, dP_dT = torch.autograd.grad(P.sum(), (rho, T), create_graph=create_graph,
                                         retain_graph=True)
    de_drho, de_dT = torch.autograd.grad(e.sum(), (rho, T), create_graph=create_graph)
    return {"P": P, "e": e, "dP_drho": dP_drho, "dP_dT": dP_dT,
            "de_drho": de_drho, "de_dT": de_dT}


def sample_collocation(n: int, device="cpu", generator: torch.Generator | None = None
                       ) -> tuple[torch.Tensor, torch.Tensor]:
    """Случайные точки для физических штрафов, равномерно по log rho и log T.

    Args:
        n: число точек.
        device: устройство torch.
        generator: генератор случайных чисел (для воспроизводимости).

    Returns:
        (rho, T), форма (n,).
    """
    u = torch.rand(n, 2, device=device, generator=generator)
    log_rho = LOG_RHO_BOUNDS[0] + u[:, 0] * (LOG_RHO_BOUNDS[1] - LOG_RHO_BOUNDS[0])
    log_T = LOG_T_BOUNDS[0] + u[:, 1] * (LOG_T_BOUNDS[1] - LOG_T_BOUNDS[0])
    return 10 ** log_rho, 10 ** log_T
