# 4.1 Класс-обёртка EOSModel (код — src/eos/inference.py): predict, sound_speed, check_physics
import importlib, json, time, torch
import eos.normalize as nz, eos.model as em, eos.pinn as pn, eos.metrics as mt, eos.baseline as bl, eos.inference as inf
for m in (nz, em, pn, mt, bl, inf):
    importlib.reload(m)
pd.set_option('display.width', 250)

MODEL = ARTIFACTS / 'pinn.pt'
NORM = ARTIFACTS / 'normalization.json'
model = inf.EOSModel(MODEL, NORM, device=DEVICE)                          # float32 — быстрый
model64 = inf.EOSModel(MODEL, NORM, device=DEVICE, dtype=torch.float64)   # float64 — для метрик
print('Конфигурация сети:', model.net.config)

# Примеры: обычный алюминий, сжатый горячий, разреженная плазма
rho_ex = np.array([2.7, 10.0, 0.01, 1000.0])
T_ex = np.array([300.0, 1e4, 1e6, 1e5])
P_ex, e_ex = model.predict(rho_ex, T_ex)
flags = model.check_physics(rho_ex, T_ex)
display(pd.DataFrame({'ρ, г/см³': rho_ex, 'T, K': T_ex, 'P, ГПа': P_ex, 'e, кДж/г': e_ex,
                      'c_s, км/с': model.sound_speed(rho_ex, T_ex),
                      **{k: v for k, v in flags.items() if k != 'cons_residual'}}))

# Проверка физики на 100 000 случайных точках всей области (ρ, T)
rho_q, T_q = bl.random_queries(100_000, seed=1)
flags = model.check_physics(rho_q, T_q)
print('Доля точек, где условие выполнено, %:')
print(pd.Series({k: 100 * v.mean() for k, v in flags.items() if k != 'cons_residual'}).round(3).to_string())
