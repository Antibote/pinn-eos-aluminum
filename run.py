"""Точка входа: уравнение состояния алюминия на PINN из командной строки.

Примеры:
    python run.py                                  # демонстрация на характерных точках
    python run.py --rho 2.7 10 --T 300 1e4         # свои точки (списки одной длины)
    python run.py --benchmark                      # время на 10^5 случайных запросов
    python run.py --evaluate                       # метрики на общем test (скачает данные)
    python run.py --model artifacts/pinn_fp16.pt   # другая версия модели
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT / "src"))

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
import torch  # noqa: E402

from eos.inference import EOSModel  # noqa: E402

DEMO = [(2.7, 300.0), (10.0, 1e4), (0.01, 1e6), (1000.0, 1e5), (1e-3, 0.1)]


def show_points(model: EOSModel, rho, T) -> None:
    """Печатает P, e, c_s и проверку термодинамики для точек (rho, T)."""
    P, e = model.predict(rho, T)
    flags = model.check_physics(rho, T)
    table = pd.DataFrame({
        "ρ, г/см³": rho, "T, K": T, "P, ГПа": P, "e, кДж/г": e,
        "c_s, км/с": model.sound_speed(rho, T),
        "∂P/∂ρ>0": flags["mechanical"], "∂e/∂T≥0": flags["thermal"], "0<c_s<c": flags["sound"],
        "в таблице": flags["in_domain"],
    })
    with pd.option_context("display.width", 200, "display.float_format", "{:.5g}".format):
        print(table.to_string(index=False))


def main() -> None:
    """Разбирает аргументы командной строки и выполняет выбранные действия."""
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--model", default=ROOT / "artifacts" / "pinn.pt", type=Path, help="чекпоинт модели")
    ap.add_argument("--norm", default=ROOT / "artifacts" / "normalization.json", type=Path, help="нормировка")
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    ap.add_argument("--rho", type=float, nargs="+", help="плотность, г/см³")
    ap.add_argument("--T", type=float, nargs="+", help="температура, K")
    ap.add_argument("--benchmark", action="store_true", help="время на 10^5 случайных запросов")
    ap.add_argument("--evaluate", action="store_true", help="метрики на общем test (нужны данные)")
    args = ap.parse_args()

    model = EOSModel(args.model, args.norm, device=args.device)
    print(f"Модель {args.model.name}: {model.net.config}, устройство {args.device}\n")

    if args.rho or args.T:
        if not (args.rho and args.T) or len(args.rho) != len(args.T):
            ap.error("--rho и --T задаются вместе и одной длины")
        show_points(model, np.array(args.rho), np.array(args.T))
    elif not (args.benchmark or args.evaluate):
        rho, T = map(np.array, zip(*DEMO))
        show_points(model, rho, T)

    if args.benchmark:
        from eos.baseline import benchmark
        print(f"\nВремя на 10^5 случайных запросов ({args.device}, float32): {benchmark(model, n=100_000):.4f} с")

    if args.evaluate:
        from eos.data import download, load_all
        from eos.metrics import evaluate
        df = load_all(download())
        test_df = df.iloc[np.load(ROOT / "artifacts" / "split.npz")["test"]]
        model64 = EOSModel(args.model, args.norm, device=args.device, dtype=torch.float64)
        metrics, _ = evaluate(model64.predict, test_df, args.model.name)
        print("\nМетрики на test (ошибки в %, нарушения — доля точек в %):")
        print(pd.Series(metrics).to_string())


if __name__ == "__main__":
    main()
