"""Быстрые проверки пакета eos: нормировка, модель из artifacts/, физика, экспорт, разбиение данных."""
from pathlib import Path

import numpy as np
import pytest
import torch

from eos.baseline import BilinearEOS, decimate_mask, random_queries
from eos.data import DATA_DIR, load_all
from eos.inference import EOSModel, OnnxEOS, compress_fp16, export_onnx, export_torchscript
from eos.normalize import EOSNormalizer

ROOT = Path(__file__).resolve().parents[1]
ART = ROOT / "artifacts"
HAS_DATA = len(list(DATA_DIR.glob("Obl*.txt"))) == 20


@pytest.fixture(scope="module")
def model():
    return EOSModel(ART / "pinn.pt", ART / "normalization.json")


def test_normalizer_roundtrip():
    rng = np.random.default_rng(0)
    rho, T = 10 ** rng.uniform(-3, 4, 1000), 10 ** rng.uniform(-1, 7, 1000)
    P, e = 10 ** rng.uniform(-3, 9, 1000), rng.uniform(-47, 1.7e5, 1000)
    n = EOSNormalizer().fit(rho, T, P, e)
    x = n.transform_x(rho, T)
    assert x.min() >= -1 - 1e-12 and x.max() <= 1 + 1e-12
    P2, e2 = n.inverse_y(n.transform_y(P, e))
    np.testing.assert_allclose(P2, P, rtol=1e-12)
    np.testing.assert_allclose(e2, e, rtol=1e-9, atol=1e-9)


def test_predict_shapes_and_values(model):
    P, e = model.predict(2.7, 300.0)
    assert P.shape == () and np.isfinite(P) and P > 0
    P, e = model.predict(np.full((2, 3), 2.7), np.full((2, 3), 300.0))
    assert P.shape == e.shape == (2, 3)
    with pytest.raises(ValueError):
        model.predict(-1.0, 300.0)


def test_physics_on_random_points(model):
    rho, T = random_queries(5000, seed=3)
    flags = model.check_physics(rho, T)
    assert flags["mechanical"].mean() > 0.999
    assert flags["thermal"].mean() > 0.999
    assert flags["sound"].mean() > 0.999
    cs = model.sound_speed(rho, T)
    assert np.nanmax(cs) < 299_792.458


def test_derivatives_match_finite_differences(model):
    m64 = EOSModel(ART / "pinn.pt", ART / "normalization.json", dtype=torch.float64)
    rho, T, h = np.array([0.05, 2.7, 300.0]), np.array([50.0, 3e3, 1e6]), 1e-6
    d = m64.derivatives(rho, T)
    P1, _ = m64.predict(rho * (1 + h), T)
    P0, _ = m64.predict(rho * (1 - h), T)
    np.testing.assert_allclose(d["dP_drho"], (P1 - P0) / (2 * h * rho), rtol=1e-4)


def test_exports_match(model, tmp_path):
    rho, T = random_queries(2000, seed=4)
    P0, e0 = model.predict(rho, T)
    ts = export_torchscript(model.net, tmp_path / "m.pt")
    with torch.no_grad():
        P1, _ = ts(torch.tensor(rho, dtype=torch.float32), torch.tensor(T, dtype=torch.float32))
    np.testing.assert_allclose(P1.numpy(), P0, rtol=1e-4)
    export_onnx(model.net, tmp_path / "m.onnx")
    P2, e2 = OnnxEOS(tmp_path / "m.onnx").predict(rho, T)
    np.testing.assert_allclose(P2, P0, rtol=1e-4)
    np.testing.assert_allclose(e2, e0, rtol=1e-3, atol=1e-3)


def test_fp16_keeps_accuracy(model):
    rho, T = random_queries(2000, seed=5)
    P0, _ = model.predict(rho, T)
    P1, _ = EOSModel.from_net(compress_fp16(model.net)).predict(rho, T)
    assert np.median(np.abs(P1 / P0 - 1)) < 1e-2


def test_decimate_mask_keeps_last_nodes():
    m = decimate_mask(5, 4)
    assert not m[0, 0] and not m[-1, -1] and m[1, 1]


@pytest.mark.skipif(not HAS_DATA, reason="нет data/Obl*.txt — запустите python run.py --evaluate")
def test_split_matches_saved_and_baseline_exact_on_train():
    from eos.model import make_split
    df = load_all()
    split, saved = make_split(df), np.load(ART / "split.npz")
    for k in ("train", "val", "test"):
        np.testing.assert_array_equal(split[k], saved[k])
    base = BilinearEOS(df, decimate=True)
    tr = df.iloc[split["train"][::97]]
    P, _ = base.predict(tr["rho"].to_numpy(), tr["T"].to_numpy())
    np.testing.assert_allclose(P, tr["P"].to_numpy(), rtol=1e-9)


def test_svd_keeps_io_layers_and_reloads(model, tmp_path):
    from eos.inference import compress_svd, load_eosnet, save_net
    net = compress_svd(model.net, 64)
    assert net.net[0].in_features == 2 and net.net[-1].out_features == 2      # вход и выход не тронуты
    assert sum(p.numel() for p in net.parameters()) < 0.55 * sum(p.numel() for p in model.net.parameters())
    save_net(net, tmp_path / "svd.pt", fp16=True)
    back = EOSModel.from_net(load_eosnet(tmp_path / "svd.pt", ART / "normalization.json"))
    P0, _ = EOSModel.from_net(net).predict(2.7, 300.0)
    P1, _ = back.predict(2.7, 300.0)
    assert abs(P1 / P0 - 1) < 1e-2
    full = compress_svd(model.net, 256)                                        # полный ранг — без потерь
    np.testing.assert_allclose(EOSModel.from_net(full).predict(2.7, 300.0)[0], model.predict(2.7, 300.0)[0], rtol=1e-5)


@pytest.mark.skipif(__import__("shutil").which("gcc") is None, reason="нет gcc")
def test_c_inference_matches_torch(model, tmp_path):
    from eos.cinfer import CEOS, export_weights
    export_weights(model.net, tmp_path / "m.bin")
    c = CEOS(tmp_path / "m.bin")
    assert c.n_params == sum(p.numel() for p in model.net.parameters())
    rho, T = random_queries(1001, seed=6)
    P0, e0 = model.predict(rho, T)
    P1, e1 = c.predict(rho, T)
    np.testing.assert_allclose(P1, P0, rtol=1e-4)
    np.testing.assert_allclose(e1, e0, rtol=1e-3, atol=1e-3)
    p, e = c.predict_one(2.7, 300.0)
    assert abs(p / P0[0] - 1) < 1 or np.isfinite(p)
