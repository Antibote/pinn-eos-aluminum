# 3.1 Физически ориентированная функция потерь (код — src/eos/pinn.py)
import importlib, json, torch
import eos.normalize as nz, eos.model as em, eos.pinn as pn, eos.metrics as mt, eos.baseline as bl, eos.inference as inf
for m in (nz, em, mt, pn, bl, inf):
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

# Функция потерь PINN (производные — autograd по ln ρ и ln T, штрафы безразмерные):
#   L = MSE(данные)                                     — log10 P и asinh e после RobustScaler
#     + λ_ρ · mean max(0, −∂lnP/∂lnρ)                  — механическая устойчивость (∂P/∂ρ)_T > 0
#     + λ_T · mean max(0, −∂e/∂lnT) / (|e| + 1)         — термическая устойчивость (∂e/∂T)_ρ ≥ 0
#     + λ_c · mean нарушения 0 < c_s < c                 — скорость звука (3.3)
# Штрафы считаются в n_coll случайных точках по всей области (log ρ, log T); λ растут первые 20% шагов.
# Сеть 256×4, активация GELU (гладкая — производные сети входят в loss), градиенты — обычная сумма.
CFG = dict(hidden=256, layers=4, activation='gelu', loss='mse', grad='fixed',
           lam={'mono_rho': 1, 'mono_T': 1, 'sound': 1}, batch=8192, n_coll=1024, lr=2e-3)
STEPS = 30_000   # для пробного запуска поставьте 2000

# Штрафы на необученной сети (0 — условие выполнено во всех точках коллокации)
probe = em.EOSNet(norm, hidden=CFG['hidden'], layers=CFG['layers'], activation=CFG['activation']).to(DEVICE)
print('Штрафы на необученной сети:', {k: round(v.item(), 5) for k, v in trainer.physics_loss(probe, 4096).items()})

# Термодинамическая согласованность (∂e/∂ρ)_T = (P − T ∂P/∂T)/ρ² в loss не входит: её нарушает
# сама таблица (ratio_median = 0.5 в холодной части вместо 1), штраф спорил бы с данными
cons_table = pn.table_consistency(df)
cons_table.to_csv(ARTIFACTS / 'table_consistency.csv', index=False)
print(cons_table.to_string(index=False, float_format=lambda v: f'{v:.3g}'))
