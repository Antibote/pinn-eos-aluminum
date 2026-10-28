# 3.1 Подготовка: данные на GPU и настройки PINN (код — src/eos/pinn.py)
import importlib, json, torch
import eos.normalize as nz, eos.model as em, eos.pinn as pn, eos.metrics as mt, eos.baseline as bl
for m in (nz, em, pn, mt, bl):
    importlib.reload(m)
pd.set_option('display.width', 250)
print('Устройство:', DEVICE)   # должно быть cuda — иначе включите GPU в настройках среды

norm = nz.EOSNormalizer.load(ARTIFACTS / 'normalization.json')
split = dict(np.load(ARTIFACTS / 'split.npz'))
test_df = df.iloc[split['test']]

def to_t(a):
    return torch.tensor(np.asarray(a), dtype=torch.float32, device=DEVICE)

trainer = pn.Trainer(*(to_t(df[c].values) for c in ['rho', 'T', 'P', 'e']),
                     idx_train=torch.tensor(split['train'], device=DEVICE),
                     idx_val=torch.tensor(split['val'], device=DEVICE),
                     w_train=to_t(em.region_weights(df.iloc[split['train']])))

# Конфигурация PINN:
#   GELU — гладкая активация (производные сети входят в loss);
#   MSE — loss по данным в нормированном пространстве (log10 P, asinh e);
#   fixed — loss = данные + Σ λ·штраф, обычный backward;
#   штрафы устойчивости ∂P/∂ρ > 0 и ∂e/∂T ≥ 0 с весом 1 (λ плавно растёт первые 20% шагов)
CFG = dict(hidden=256, layers=4, activation='gelu', loss='mse', grad='fixed',
           lam={'mono_rho': 1, 'mono_T': 1}, batch=8192, n_coll=1024, lr=2e-3)
STEPS = 30_000   # для пробного запуска поставьте 2000

# Согласованность самой таблицы (∂e/∂ρ)_T = (P − T ∂P/∂T)/ρ²: ratio_median = 1 — согласовано.
# В холодной части таблицы 0.5, поэтому штраф согласованности в loss не включаем
cons_table = pn.table_consistency(df)
cons_table.to_csv(ARTIFACTS / 'table_consistency.csv', index=False)
print(cons_table.to_string(index=False, float_format=lambda v: f'{v:.3g}'))
