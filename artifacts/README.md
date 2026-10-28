# artifacts/ — результаты ноутбука

Все файлы создаются ячейками `notebooks/pinn_eos_colab.ipynb` (номер раздела — в скобках) и сохраняются в репозиторий функцией `save_to_repo`.

## Модели

| Файл | Что это |
|---|---|
| `pinn.pt` | **Итоговая PINN** 256×4, GELU: `{"state_dict", "config", "train", "steps"}` (3.2) |
| `pinn_fp16.pt` | та же PINN с весами float16 — 0,38 МБ, без потери точности (4.4) |
| `pinn.onnx`, `pinn_torchscript.pt` | экспорт PINN, нормировка встроена: вход ρ, T → выход P, e (4.3) |
| `pinn_int8w.pt` | int8-веса — для сравнения, точность хуже (4.4) |
| `pinn_128x4.pt`, `pinn_64x4.pt` | дистиллированные PINN (5.2) |
| `optimized/` | варианты из 5.3–5.4: float32 / float16 / ONNX для 256×4, 128×4, 64×4 и SVD r = 64, 32 |
| `c/` | веса для инференса на C (`pinn.bin`, `pinn_svd64_fp16.bin`) и ONNX-версии (5.5) |
| `vanilla_model.pt` | Vanilla MLP 256×4, SiLU (2.1) |
| `normalization.json` | параметры нормировки `EOSNormalizer` (1.3) |
| `split.npz` | общее разбиение train / val / test, seed 0 (1.3) |

## Метрики

| Файл | Что это |
|---|---|
| `metrics_final.csv`, `metrics_final.md` | **итоговая таблица** Baseline / Vanilla / PINN / PINN float16 (4.5) |
| `metrics_baseline.json`, `metrics_vanilla.json`, `metrics_pinn.json` | метрики каждой модели на test (1.4, 2.3, 3.4) |
| `metrics_compare.csv` | сравнение с базовой моделью из 3.4 |
| `metrics_speed.json`, `metrics_compression.csv`, `metrics_optimization.csv`, `metrics_svd.csv`, `metrics_c.csv` | скорость, сжатие, оптимизация, SVD, C (4.2, 4.4, 5.3–5.5) |
| `violations_vanilla.json`, `compare_baseline_vanilla.csv` | нарушения и сравнение из раздела 2 |
| `table_consistency.csv` | согласованность самой таблицы по областям (3.1) |
| `regions_summary.csv`, `events.csv`, `seams.csv` | проверка областей, аномалии, стыки (1.1) |

## Графики — `figures/`

`eda_*`, `eos_sections.png` — данные (1.1–1.2); `baseline_errors.png` (1.4); `vanilla/`, `vanilla_sound_speed.png` (2); `pinn_learning_curve.png`, `pinn_errors.png`, `pinn_sections.png`, `pinn_sound_speed.png` (3); `optimization.png` (5.3), `svd_compression.png` (5.4).
