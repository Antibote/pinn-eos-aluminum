"""Загрузка табличных данных уравнения состояния алюминия.

Данные — 20 файлов ``data/OblNN.txt``. Каждый файл — полная прямоугольная
сетка по (ρ, T) в одной из областей 4 × 5 (4 диапазона плотности ×
5 диапазонов температуры). Колонки: ``rho [g/cm^3], T [K], P [GPa], e [kJ/g]``.

Диапазоны в шапках полуоткрытые: [min, max). Верхняя граница — первая точка
соседней области, поэтому области стыкуются без повторов. Включена она только
у крайних областей (T = 1e7 K и rho = 1e4 г/см³).
"""

from __future__ import annotations

import re
from pathlib import Path

import numpy as np
import pandas as pd

DATA_DIR = Path(__file__).resolve().parents[2] / "data"
COLUMNS = ["rho", "T", "P", "e"]


def read_region(path: str | Path) -> tuple[pd.DataFrame, dict]:
    """Читает один файл области.

    Args:
        path: путь к файлу ``OblNN.txt``.

    Returns:
        (df, meta): таблица с колонками rho, T, P, e и словарь параметров
        из шапки файла (name, rho_range, T_range, grid).
    """
    path = Path(path)
    meta = {"name": path.stem}
    with open(path) as f:
        for line in f:
            if not line.startswith("#"):
                break
            if m := re.match(r"# rho range:\s*\[(.+),\s*(.+)\]", line):
                meta["rho_range"] = (float(m[1]), float(m[2]))
            elif m := re.match(r"# T range:\s*\[(.+),\s*(.+)\]", line):
                meta["T_range"] = (float(m[1]), float(m[2]))
            elif m := re.match(r"# Grid type:\s*(.+)", line):
                meta["grid"] = m[1].strip()
    df = pd.read_csv(path, comment="#", sep=r"\s+", header=None, names=COLUMNS)
    return df, meta


def load_all(data_dir: str | Path = DATA_DIR) -> pd.DataFrame:
    """Загружает все области в одну таблицу.

    Args:
        data_dir: папка с файлами ``Obl*.txt``.

    Returns:
        DataFrame с колонками rho, T, P, e, region (имя области, напр. "Obl07").
        Повторов точек (rho, T) между областями нет.

    Raises:
        FileNotFoundError: если в папке нет ни одного файла ``Obl*.txt``.
    """
    files = sorted(Path(data_dir).glob("Obl*.txt"))
    if not files:
        raise FileNotFoundError(f"Нет файлов Obl*.txt в {data_dir}")
    parts = []
    for f in files:
        df, meta = read_region(f)
        df["region"] = meta["name"]
        parts.append(df)
    return pd.concat(parts, ignore_index=True)


def region_grid(df: pd.DataFrame) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Превращает таблицу одной области в двумерную сетку.

    Args:
        df: строки одной области (полная сетка rho × T).

    Returns:
        (rho, T, P, e): rho формы (Nr,), T формы (Nt,), P и e формы (Nt, Nr),
        где P[i, j] = P(T[i], rho[j]).

    Raises:
        ValueError: если точки не образуют полную сетку.
    """
    rho = np.unique(df["rho"].to_numpy())
    T = np.unique(df["T"].to_numpy())
    if len(rho) * len(T) != len(df):
        raise ValueError("Точки области не образуют полную сетку rho × T")
    s = df.sort_values(["T", "rho"])
    P = s["P"].to_numpy().reshape(len(T), len(rho))
    e = s["e"].to_numpy().reshape(len(T), len(rho))
    return rho, T, P, e
