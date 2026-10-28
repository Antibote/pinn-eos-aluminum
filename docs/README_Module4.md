# Модуль 4. Инференс, упаковка и сжатие

**Ответственный:** Володченко Александра · **Ноутбук:** раздел 4 (или `notebooks/section4_inference.ipynb`) · **Код:** `src/eos/inference.py`

## Класс EOSModel

```python
from eos import EOSModel
model = EOSModel("artifacts/pinn.pt", "artifacts/normalization.json", device="cpu")
P, e = model.predict(rho, T)          # ГПа, кДж/г (числа или массивы)
d = model.derivatives(rho, T)         # dP_drho, dP_dT, de_drho, de_dT — autograd, float64
cs = model.sound_speed(rho, T)        # км/с
flags = model.check_physics(rho, T)   # mechanical, thermal, sound, consistent, in_domain
```

## Размер и скорость (10⁵ запросов)

| | время, с |
|---|---|
| PINN, GPU T4, float32 | 0,023 |
| PINN, CPU (Colab), float32 | 0,78 |
| бейзлайн, CPU | 0,06 |

Файл `pinn.pt` — 0,76 МБ (лимит 10 МБ).

## Экспорт

TorchScript (`pinn_torchscript.pt`, 0,78 МБ) и ONNX (`pinn.onnx`, 0,76 МБ); нормировка зашита в граф — вход ρ, T, выход P, e. Отличие от исходной модели ≤ 5·10⁻⁵. ONNX Runtime на CPU — 0,52 с (в 1,5 раза быстрее PyTorch). В ONNX нет `Sinh`, поэтому обратное преобразование энергии записано через `exp`.

## Сжатие

| | размер | ошибка P / e (медиана) |
|---|---|---|
| float32 | 0,76 МБ | 0,23% / 0,25% |
| **float16** | **0,38 МБ** | **0,23% / 0,25%** |
| int8, только веса | 0,20 МБ | 1,28% / 4,28% |
| int8, динамическое | 0,20 МБ | 10,7% / 27,9% |

float16 — без потерь; int8 непригодно (выходы сети — логарифмы, малая погрешность становится большой относительной ошибкой).

**Артефакты:** `pinn_fp16.pt`, `pinn_int8w.pt`, `pinn_torchscript.pt`, `pinn.onnx`, `metrics_speed.json`, `metrics_compression.csv`, `metrics_final.csv/.md`.
