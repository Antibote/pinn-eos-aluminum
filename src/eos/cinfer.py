"""Инференс PINN на C: экспорт весов, сборка библиотеки ``csrc/eos_pinn.c``, обёртка ``CEOS``.

На C перенесён только прямой проход уже обученной сети (нормировка → слои Linear + GELU →
обратная нормировка). Обучение и производные через autograd остаются в PyTorch.
Библиотека не зависит от Python и PyTorch: тот же файл весов и ``eos_pinn.c`` подключаются
в расчётный код на C/C++/Fortran (пример — ``csrc/example.c``).
"""

from __future__ import annotations

import ctypes
import struct
import subprocess
from pathlib import Path

import numpy as np
import torch

from eos.model import EOSNet

CSRC = Path(__file__).resolve().parents[2] / "csrc"
_ACT = {"gelu": 1, "silu": 2, "tanh": 3}
_DBL = np.ctypeslib.ndpointer(dtype=np.float64, flags="C_CONTIGUOUS")


def export_weights(net: EOSNet, path: str | Path) -> float:
    """Пишет веса и нормировку сети в бинарный файл для ``eos_load`` (формат — в eos_pinn.c).

    Поддерживаются обычная ``EOSNet`` и SVD-сжатая (``inference.compress_svd``): пара слоёв
    h→r→h без активации между ними записывается как два слоя.

    Returns:
        Размер файла, МБ.
    """
    act = _ACT[net.config["activation"]]
    layers = []  # (W, b, активация после)

    def walk(mod):
        for m in mod:
            if isinstance(m, torch.nn.Sequential):
                walk(m)
            elif isinstance(m, torch.nn.Linear):
                layers.append([m.weight.detach().cpu().float().numpy(),
                               None if m.bias is None else m.bias.detach().cpu().float().numpy(), 0])
            else:  # активация — относится к последнему линейному слою
                layers[-1][2] = 1

    walk(net.net)
    n = net.norm
    with open(path, "wb") as f:
        f.write(b"EOSP" + struct.pack("<3i", 1, act, len(layers)))
        f.write(struct.pack("<9d", *n.x_min, *n.x_max, *n.y_center, *n.y_scale, n.E0))
        for W, b, a in layers:
            f.write(struct.pack("<4i", W.shape[1], W.shape[0], b is not None, a))
            f.write(np.ascontiguousarray(W, dtype="<f4").tobytes())
            if b is not None:
                f.write(np.ascontiguousarray(b, dtype="<f4").tobytes())
    return Path(path).stat().st_size / 2**20


def build(out_dir: str | Path | None = None, force: bool = False) -> Path:
    """Собирает ``libeos_pinn.so`` из ``csrc/eos_pinn.c`` (gcc -O3 -march=native -ffast-math -fopenmp).

    Args:
        out_dir: куда положить библиотеку (по умолчанию ``csrc/build``).
        force: пересобрать, даже если библиотека новее исходника.

    Returns:
        Путь к библиотеке.
    """
    out_dir = Path(out_dir or CSRC / "build")
    out_dir.mkdir(parents=True, exist_ok=True)
    lib = out_dir / "libeos_pinn.so"
    src = CSRC / "eos_pinn.c"
    if force or not lib.exists() or lib.stat().st_mtime < src.stat().st_mtime:
        subprocess.run(["gcc", "-O3", "-march=native", "-ffast-math", "-fopenmp", "-fPIC", "-shared",
                        str(src), "-o", str(lib), "-lm"], check=True)
    return lib


class CEOS:
    """Модель на C с тем же ``predict(rho, T)``, что у ``EOSModel`` (для метрик и замеров).

    Args:
        weights: файл из ``export_weights``.
        threads: потоки OpenMP для пакетного режима (None — все ядра).
        lib: путь к библиотеке (по умолчанию собирается ``build()``).
    """

    def __init__(self, weights: str | Path, threads: int | None = None, lib: str | Path | None = None):
        self.lib = ctypes.CDLL(str(lib or build()))
        self.lib.eos_load.restype = ctypes.c_void_p
        self.lib.eos_load.argtypes = [ctypes.c_char_p]
        self.lib.eos_free.argtypes = [ctypes.c_void_p]
        self.lib.eos_n_params.restype = ctypes.c_long
        self.lib.eos_n_params.argtypes = [ctypes.c_void_p]
        self.lib.eos_set_threads.argtypes = [ctypes.c_int]
        self.lib.eos_predict.argtypes = [ctypes.c_void_p, ctypes.c_long, _DBL, _DBL, _DBL, _DBL]
        self.lib.eos_predict_one.argtypes = [ctypes.c_void_p, ctypes.c_double, ctypes.c_double,
                                             ctypes.POINTER(ctypes.c_double), ctypes.POINTER(ctypes.c_double)]
        self.handle = self.lib.eos_load(str(weights).encode())
        if not self.handle:
            raise RuntimeError(f"не удалось загрузить веса {weights}")
        self.lib.eos_set_threads(threads or 0)
        self._p, self._e = ctypes.c_double(), ctypes.c_double()

    @property
    def n_params(self) -> int:
        """Число параметров загруженной сети."""
        return self.lib.eos_n_params(self.handle)

    def predict(self, rho, T):
        """(ρ [г/см³], T [K]) → (P [ГПа], e [кДж/г]) для массивов; считается на C."""
        rho = np.ascontiguousarray(np.ravel(rho), dtype=np.float64)
        T = np.ascontiguousarray(np.ravel(T), dtype=np.float64)
        if np.any(rho <= 0) or np.any(T <= 0):
            raise ValueError("rho и T должны быть > 0")
        P, e = np.empty_like(rho), np.empty_like(rho)
        self.lib.eos_predict(self.handle, len(rho), rho, T, P, e)
        return P, e

    def predict_one(self, rho: float, T: float) -> tuple[float, float]:
        """Одна точка без накладных расходов numpy — для замера задержки одиночного запроса."""
        self.lib.eos_predict_one(self.handle, rho, T, ctypes.byref(self._p), ctypes.byref(self._e))
        return self._p.value, self._e.value

    def __del__(self):
        if getattr(self, "handle", None):
            self.lib.eos_free(self.handle)
            self.handle = None
