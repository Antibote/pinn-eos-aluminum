"""Нормировка входов и выходов сети (sklearn.preprocessing).

- плотность и температура: log10, затем MinMaxScaler в [-1, 1];
- давление: log10 (P > 0 везде), затем RobustScaler (медиана и IQR);
- энергия: asinh(e / E0) — «логарифм со знаком», работает для e ≤ 0, затем RobustScaler.

Параметры подбираются только на train. После ``fit`` хранятся как массивы, поэтому
те же преобразования есть и для numpy (подготовка данных), и для torch
(дифференцируемо — для производных в PINN).
"""

from __future__ import annotations

import json

import numpy as np
from sklearn.preprocessing import MinMaxScaler, RobustScaler


class EOSNormalizer:
    """Нормализация: (rho, T) -> log10 -> MinMax[-1,1]; P -> log10, e -> asinh(e/E0) -> RobustScaler.

    Args:
        E0: масштаб энергии в asinh(e / E0), кДж/г.
    """

    def __init__(self, E0: float = 1.0):
        self.E0 = float(E0)
        self.x_min = self.x_max = self.y_center = self.y_scale = None

    @staticmethod
    def _check_positive(name, a):
        if np.any(np.asarray(a) <= 0):
            raise ValueError(f'{name} должно быть > 0')

    def _log_x(self, rho, T):
        self._check_positive('rho', rho)
        self._check_positive('T', T)
        return np.column_stack([np.log10(rho), np.log10(T)])

    def _raw_y(self, P, e):
        self._check_positive('P', P)
        return np.column_stack([np.log10(P), np.arcsinh(np.asarray(e) / self.E0)])

    def fit(self, rho, T, P, e):
        """Подбор параметров (только на train)."""
        xs = MinMaxScaler(feature_range=(-1, 1)).fit(self._log_x(rho, T))
        ys = RobustScaler().fit(self._raw_y(P, e))
        self.x_min, self.x_max = xs.data_min_.astype(float), xs.data_max_.astype(float)
        self.y_center, self.y_scale = ys.center_.astype(float), ys.scale_.astype(float)
        return self

    def transform_x(self, rho, T):
        """(rho [г/см³], T [K]) -> вход сети (N, 2)."""
        return 2.0 * (self._log_x(rho, T) - self.x_min) / (self.x_max - self.x_min) - 1.0

    def transform_y(self, P, e):
        """(P [ГПа], e [кДж/г]) -> цели сети (N, 2)."""
        return (self._raw_y(P, e) - self.y_center) / self.y_scale

    def inverse_y(self, y):
        """Выход сети (N, 2) -> (P [ГПа], e [кДж/г])."""
        raw = np.asarray(y) * self.y_scale + self.y_center
        return 10.0 ** raw[:, 0], self.E0 * np.sinh(raw[:, 1])

    def in_domain(self, rho, T):
        """Булев массив: точка внутри домена обучения."""
        lx = self._log_x(rho, T)
        return np.all((lx >= self.x_min) & (lx <= self.x_max), axis=1)

    def torch_x(self, rho, T):
        """Дифференцируемый аналог transform_x для torch-тензоров (N,)."""
        import torch
        x_min = torch.as_tensor(self.x_min, dtype=rho.dtype, device=rho.device)
        x_max = torch.as_tensor(self.x_max, dtype=rho.dtype, device=rho.device)
        lx = torch.stack([torch.log10(rho), torch.log10(T)], dim=-1)
        return 2.0 * (lx - x_min) / (x_max - x_min) - 1.0

    def torch_y(self, P, e):
        """Дифференцируемый аналог transform_y для torch-тензоров (N,)."""
        import torch
        c = torch.as_tensor(self.y_center, dtype=P.dtype, device=P.device)
        s = torch.as_tensor(self.y_scale, dtype=P.dtype, device=P.device)
        raw = torch.stack([torch.log10(P), torch.asinh(e / self.E0)], dim=-1)
        return (raw - c) / s

    def torch_inverse_y(self, y):
        """Дифференцируемый аналог inverse_y для torch-тензора (N, 2)."""
        import torch
        c = torch.as_tensor(self.y_center, dtype=y.dtype, device=y.device)
        s = torch.as_tensor(self.y_scale, dtype=y.dtype, device=y.device)
        raw = y * s + c
        return torch.pow(10.0, raw[:, 0]), self.E0 * torch.sinh(raw[:, 1])

    def save(self, path='normalization.json'):
        """Сохранение параметров в JSON."""
        d = {'x_cols': ['log10_rho', 'log10_T'], 'x_min': list(self.x_min), 'x_max': list(self.x_max),
             'x_range': [-1.0, 1.0], 'y_cols': ['log10_P', 'asinh_e_over_E0'],
             'y_center': list(self.y_center), 'y_scale': list(self.y_scale), 'E0': self.E0,
             'units': {'rho': 'g/cm^3', 'T': 'K', 'P': 'GPa', 'e': 'kJ/g'}}
        with open(path, 'w', encoding='utf-8') as f:
            json.dump(d, f, indent=2, ensure_ascii=False)

    @classmethod
    def load(cls, path='normalization.json'):
        """Загрузка параметров из JSON."""
        with open(path, encoding='utf-8') as f:
            d = json.load(f)
        n = cls(E0=d['E0'])
        n.x_min, n.x_max = np.array(d['x_min']), np.array(d['x_max'])
        n.y_center, n.y_scale = np.array(d['y_center']), np.array(d['y_scale'])
        return n
