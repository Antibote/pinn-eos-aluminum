"""Инференс: класс ``EOSModel``, экспорт (TorchScript, ONNX) и сжатие модели.

``EOSModel`` — то, чем пользуется внешний код: физические ρ [г/см³] и T [K] на входе,
P [ГПа], e [кДж/г], скорость звука и проверка термодинамики на выходе.
"""

from __future__ import annotations

import copy
import io
from pathlib import Path

import numpy as np
import torch

from eos.model import EOSNet
from eos.normalize import EOSNormalizer
from eos.pinn import C_LIGHT_KMS, log_derivatives, physics_terms, sound_speed2


def load_eosnet(model_path: str | Path, norm_path: str | Path, device="cpu") -> EOSNet:
    """Собирает ``EOSNet`` из чекпоинта раздела 3 (``pinn.pt``) и ``normalization.json``.

    Args:
        model_path: файл {"state_dict", "config", ...} из ``torch.save`` в разделе 3.
        norm_path: параметры нормировки.
        device: куда загрузить модель.

    Returns:
        ``EOSNet`` в режиме eval.
    """
    norm = EOSNormalizer.load(norm_path)
    ckpt = torch.load(model_path, map_location=device, weights_only=False)
    model = EOSNet(norm, **ckpt["config"])
    model.load_state_dict(ckpt["state_dict"])
    return model.to(device).eval()


def state_dict_size_mb(model: torch.nn.Module) -> float:
    """Размер сохранённых весов модели (``torch.save(state_dict)``), МБ."""
    buf = io.BytesIO()
    torch.save(model.state_dict(), buf)
    return buf.getbuffer().nbytes / 2**20


class EOSModel:
    """Уравнение состояния алюминия на нейросети: P(ρ, T), e(ρ, T) и их производные.

    Args:
        model_path: чекпоинт PINN (``artifacts/pinn.pt``).
        norm_path: нормировка (``artifacts/normalization.json``).
        device: "cpu" или "cuda".
        dtype: точность predict. float32 — быстрее; float64 — для расчётов, где берутся
            конечные разности (во float32 шум округления даёт ложные «нарушения» знака производных).
        chunk: сколько точек считать за раз (ограничивает память).

    Example:
        >>> model = EOSModel("artifacts/pinn.pt", "artifacts/normalization.json")
        >>> P, e = model.predict([2.7, 10.0], [300.0, 1e4])
        >>> cs = model.sound_speed(2.7, 300.0)
        >>> flags = model.check_physics(rho, T)
    """

    def __init__(self, model_path="artifacts/pinn.pt", norm_path="artifacts/normalization.json",
                 device="cpu", dtype=torch.float32, chunk: int = 262_144):
        self.device, self.dtype, self.chunk = torch.device(device), dtype, chunk
        self.net = load_eosnet(model_path, norm_path, self.device).to(dtype)
        self.norm = self.net.norm
        self._net64 = None  # копия в float64 для производных — создаётся при первом обращении

    @classmethod
    def from_net(cls, net: EOSNet, device="cpu", dtype=torch.float32, chunk: int = 262_144) -> "EOSModel":
        """``EOSModel`` вокруг уже созданной сети (например, сжатой)."""
        obj = cls.__new__(cls)
        obj.device, obj.dtype, obj.chunk = torch.device(device), dtype, chunk
        if any(True for _ in net.parameters()):  # у квантованной сети обычных параметров нет
            net = copy.deepcopy(net).to(device=device, dtype=dtype)
        obj.net, obj.norm, obj._net64 = net.eval(), net.norm, None
        return obj

    @staticmethod
    def _as_arrays(rho, T):
        rho, T = np.broadcast_arrays(np.asarray(rho, dtype=np.float64), np.asarray(T, dtype=np.float64))
        if np.any(rho <= 0) or np.any(T <= 0):
            raise ValueError("rho и T должны быть > 0")
        return rho.ravel(), T.ravel(), rho.shape

    def in_domain(self, rho, T) -> np.ndarray:
        """True — точка внутри области таблицы (ρ 1e-3…1e4 г/см³, T 0.1…1e7 K); вне — экстраполяция."""
        rho, T, shape = self._as_arrays(rho, T)
        return self.norm.in_domain(rho, T).reshape(shape)

    @torch.no_grad()
    def predict(self, rho, T) -> tuple[np.ndarray, np.ndarray]:
        """Давление и удельная внутренняя энергия.

        Args:
            rho: плотность, г/см³ (число или массив).
            T: температура, K (число или массив той же формы).

        Returns:
            (P, e): P в ГПа, e в кДж/г, массивы формы входа.

        Raises:
            ValueError: если rho или T ≤ 0.
        """
        rho, T, shape = self._as_arrays(rho, T)
        dtype = self.dtype
        P, e = np.empty_like(rho), np.empty_like(rho)
        for i in range(0, len(rho), self.chunk):
            r = torch.as_tensor(rho[i:i + self.chunk], dtype=dtype, device=self.device)
            t = torch.as_tensor(T[i:i + self.chunk], dtype=dtype, device=self.device)
            p_, e_ = self.net(r, t)
            P[i:i + self.chunk], e[i:i + self.chunk] = p_.cpu().numpy(), e_.cpu().numpy()
        return P.reshape(shape), e.reshape(shape)

    __call__ = predict

    def _log_derivatives(self, rho, T):
        """P, e и производные по ln ρ, ln T (float64, autograd), по кускам."""
        if self._net64 is None:
            self._net64 = copy.deepcopy(self.net).double().eval()
        rho, T, shape = self._as_arrays(rho, T)
        out = {}
        for i in range(0, len(rho), self.chunk // 4):
            r = torch.as_tensor(rho[i:i + self.chunk // 4], device=self.device)
            t = torch.as_tensor(T[i:i + self.chunk // 4], device=self.device)
            d = log_derivatives(self._net64, r, t, create_graph=False)
            d["cons"] = physics_terms(d, r, self.norm.E0)["cons"]
            d["cs2"] = sound_speed2(d, r)
            for k, v in d.items():
                out.setdefault(k, []).append(v.detach().cpu().numpy())
        return {k: np.concatenate(v) for k, v in out.items()}, rho, T, shape

    def derivatives(self, rho, T) -> dict[str, np.ndarray]:
        """Частные производные в физических единицах (autograd, float64).

        Returns:
            dP_drho [ГПа·см³/г], dP_dT [ГПа/K], de_drho [кДж·см³/г²], de_dT [кДж/(г·K)] — формы входа.
        """
        d, rho, T, shape = self._log_derivatives(rho, T)
        res = {"dP_drho": d["P"] * d["a"] / rho, "dP_dT": d["P"] * d["g"] / T,
               "de_drho": d["e_r"] / rho, "de_dT": d["e_t"] / T}
        return {k: v.reshape(shape) for k, v in res.items()}

    def sound_speed(self, rho, T) -> np.ndarray:
        """Адиабатическая скорость звука c_s, км/с.

        c_s² = (∂P/∂ρ)_T + T (∂P/∂T)_ρ² / (ρ² (∂e/∂T)_ρ). Где c_s² ≤ 0 — NaN.
        """
        d, _, _, shape = self._log_derivatives(rho, T)
        cs2 = d["cs2"]
        return np.sqrt(np.where(cs2 > 0, cs2, np.nan)).reshape(shape)

    def check_physics(self, rho, T, cons_tol: float = 0.1) -> dict[str, np.ndarray]:
        """Проверка термодинамики в каждой точке (True — условие выполнено).

        Args:
            rho, T: точки (числа или массивы).
            cons_tol: допуск на относительную невязку согласованности.

        Returns:
            mechanical — (∂P/∂ρ)_T > 0; thermal — (∂e/∂T)_ρ ≥ 0; sound — 0 < c_s < c;
            consistent — |невязка (∂e/∂ρ)_T = (P − T ∂P/∂T)/ρ²| ≤ cons_tol;
            in_domain — точка внутри таблицы; cons_residual — сама невязка.
        """
        d, rho, T, shape = self._log_derivatives(rho, T)
        cs2 = d["cs2"]
        res = {"mechanical": d["a"] > 0, "thermal": d["e_t"] >= 0,
               "sound": (cs2 > 0) & (cs2 < C_LIGHT_KMS ** 2),
               "consistent": np.abs(d["cons"]) <= cons_tol,
               "in_domain": self.norm.in_domain(rho, T), "cons_residual": d["cons"]}
        return {k: v.reshape(shape) for k, v in res.items()}


class _ExportNet(torch.nn.Module):
    """``EOSNet`` для экспорта: нормировка — константы графа, sinh — через exp (в ONNX нет Sinh)."""

    def __init__(self, net: EOSNet):
        super().__init__()
        self.net = copy.deepcopy(net.net).cpu().float().eval()
        n = net.norm
        for name in ("x_min", "x_max", "y_center", "y_scale"):
            self.register_buffer(name, torch.tensor(getattr(n, name), dtype=torch.float32))
        self.E0 = float(n.E0)

    def forward(self, rho: torch.Tensor, T: torch.Tensor):
        lx = torch.stack([torch.log10(rho), torch.log10(T)], dim=-1)
        y = self.net(2.0 * (lx - self.x_min) / (self.x_max - self.x_min) - 1.0)
        raw = y * self.y_scale + self.y_center
        a = raw[:, 1]
        return torch.pow(10.0, raw[:, 0]), self.E0 * 0.5 * (torch.exp(a) - torch.exp(-a))


def export_torchscript(net: EOSNet, path: str | Path) -> torch.jit.ScriptModule:
    """Сохраняет модель в TorchScript (вход — ρ, T в физических единицах, выход — P, e).

    Файл загружается без нашего кода: ``torch.jit.load(path)(rho, T)``.
    """
    example = (torch.tensor([1.0, 2.7]), torch.tensor([300.0, 1e4]))
    ts = torch.jit.trace(_ExportNet(net).eval(), example)
    ts.save(str(path))
    return ts


def export_onnx(net: EOSNet, path: str | Path, opset: int = 17) -> None:
    """Сохраняет модель в ONNX: входы rho, T (float32, длина N), выходы P, e.

    Нормировка зашита внутрь графа, так что ONNX Runtime получает физические величины.
    """
    example = (torch.tensor([1.0, 2.7]), torch.tensor([300.0, 1e4]))
    torch.onnx.export(_ExportNet(net).eval(), example, str(path), input_names=["rho", "T"], output_names=["P", "e"],
                      dynamic_axes={"rho": {0: "n"}, "T": {0: "n"}, "P": {0: "n"}, "e": {0: "n"}},
                      opset_version=opset, dynamo=False)


class OnnxEOS:
    """Инференс через ONNX Runtime с тем же ``predict(rho, T)``, что у ``EOSModel``."""

    def __init__(self, path: str | Path, providers=None):
        import onnxruntime as ort
        self.sess = ort.InferenceSession(str(path), providers=providers or ["CPUExecutionProvider"])

    def predict(self, rho, T):
        rho = np.ascontiguousarray(np.ravel(rho), dtype=np.float32)
        T = np.ascontiguousarray(np.ravel(T), dtype=np.float32)
        P, e = self.sess.run(None, {"rho": rho, "T": T})
        return P.astype(np.float64), e.astype(np.float64)


def compress_fp16(net: EOSNet) -> EOSNet:
    """Веса, округлённые до float16 (хранить вдвое меньше), считать — во float32/64."""
    small = copy.deepcopy(net).cpu().float()
    with torch.no_grad():
        for p in small.parameters():
            p.copy_(p.half().float())
    return small


def save_fp16(net: EOSNet, path: str | Path) -> float:
    """Сохраняет чекпоинт с весами float16 (формат как у ``pinn.pt``), возвращает размер файла, МБ.

    ``EOSModel(path, ...)`` открывает его так же, как обычный: веса переводятся во float32 при загрузке.
    """
    state = {k: v.detach().cpu().half() for k, v in net.state_dict().items()}
    torch.save({"state_dict": state, "config": net.config}, str(path))
    return Path(path).stat().st_size / 2**20


def compress_int8(net: EOSNet) -> torch.nn.Module:
    """Динамическая квантизация Linear-слоёв в int8 (только CPU)."""
    return torch.ao.quantization.quantize_dynamic(copy.deepcopy(net).cpu().float(),
                                                  {torch.nn.Linear}, dtype=torch.qint8)


def compress_int8_weights(net: EOSNet) -> tuple[EOSNet, dict]:
    """Int8 только для весов: каждая строка матрицы Linear — int8 со своим масштабом.

    Считается во float (веса восстанавливаются при загрузке), поэтому промежуточные значения
    не округляются — в отличие от ``compress_int8``.

    Returns:
        (net, packed): сеть с восстановленными весами для оценки точности и словарь для
        сохранения ({имя: int8-тензор, имя + ".scale": float16-масштабы, смещения — float16}).
    """
    small = copy.deepcopy(net).cpu().float()
    packed = {}
    with torch.no_grad():
        for name, p in small.named_parameters():
            if p.ndim == 2:
                scale = p.abs().amax(dim=1, keepdim=True).clamp_min(1e-12) / 127
                q = torch.round(p / scale).to(torch.int8)
                p.copy_(q.float() * scale.half().float())
                packed[name], packed[name + ".scale"] = q, scale.half()
            else:
                p.copy_(p.half().float())
                packed[name] = p.half()
    return small, packed


def save_int8_weights(net: EOSNet, packed: dict, path: str | Path) -> float:
    """Сохраняет результат ``compress_int8_weights``, возвращает размер файла, МБ."""
    torch.save({"packed": packed, "config": net.config}, str(path))
    return Path(path).stat().st_size / 2**20


def load_int8_weights(path: str | Path, norm_path: str | Path, device="cpu") -> EOSNet:
    """Загружает файл ``save_int8_weights`` в обычную ``EOSNet``."""
    ckpt = torch.load(path, map_location="cpu", weights_only=False)
    net = EOSNet(EOSNormalizer.load(norm_path), **ckpt["config"])
    pk = ckpt["packed"]
    state = {k: (pk[k].float() * pk[k + ".scale"].float()) if pk[k].dtype == torch.int8 else pk[k].float()
             for k in net.state_dict()}
    net.load_state_dict(state)
    return net.to(device).eval()


class GpuFp16EOS:
    """Инференс на GPU со смешанной точностью: умножения матриц во float16 (тензорные ядра),
    log10, pow и sinh — во float32 (``torch.autocast``). Тот же ``predict(rho, T)``.
    """

    def __init__(self, net: EOSNet, chunk: int = 262_144):
        self.net = copy.deepcopy(net).to("cuda").float().eval()
        self.chunk = chunk

    @torch.no_grad()
    def predict(self, rho, T):
        rho = np.ravel(np.asarray(rho, dtype=np.float32))
        T = np.ravel(np.asarray(T, dtype=np.float32))
        P, e = np.empty(len(rho)), np.empty(len(rho))
        for i in range(0, len(rho), self.chunk):
            r = torch.as_tensor(rho[i:i + self.chunk], device="cuda")
            t = torch.as_tensor(T[i:i + self.chunk], device="cuda")
            with torch.autocast("cuda", dtype=torch.float16):
                p_, e_ = self.net(r, t)
            P[i:i + self.chunk], e[i:i + self.chunk] = p_.float().cpu().numpy(), e_.float().cpu().numpy()
        return P, e


def save_net(net: EOSNet, path: str | Path, fp16: bool = False) -> float:
    """Сохраняет сеть в формате ``pinn.pt`` ({"state_dict", "config"}), возвращает размер файла, МБ.

    fp16=True — веса во float16 (вдвое меньше); ``load_eosnet`` / ``EOSModel`` читают оба варианта.
    """
    state = {k: (v.detach().cpu().half() if fp16 else v.detach().cpu()) for k, v in net.state_dict().items()}
    torch.save({"state_dict": state, "config": net.config}, str(path))
    return Path(path).stat().st_size / 2**20
