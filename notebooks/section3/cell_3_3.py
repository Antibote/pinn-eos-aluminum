# 3.3 Ограничение на скорость звука 0 < c_s < c (c — скорость света)
#   c_s² = (∂P/∂ρ)_T + T (∂P/∂T)_ρ² / (ρ² (∂e/∂T)_ρ);  в кДж/г = (км/с)²
# У PINN производные точные (autograd), у таблицы и бейзлайна — конечные разности
from eos.metrics import C_LIGHT_KMS
model_pinn = inf.EOSModel.from_net(pinn, device=DEVICE)

# Доля нарушений в 100 000 случайных точках всей области
rho_q, T_q = bl.random_queries(100_000, seed=1)
d = model_pinn._log_derivatives(rho_q, T_q)[0]
cs2_pinn = d['cs2']
baseline_full = bl.BilinearEOS(df)
viol_base = mt.physics_violations(baseline_full.predict, rho_q, T_q)['viol_cs_%']
print(f"Нарушение 0 < c_s < c: PINN {100 * np.mean(~((cs2_pinn > 0) & (cs2_pinn < C_LIGHT_KMS ** 2))):.4f}%, "
      f"бейзлайн (таблица) {viol_base:.4f}%")

# Карта c_s на плоскости (ρ, T) и срезы c_s(ρ) при фиксированных T
lr, lt = np.meshgrid(np.linspace(-3, 4, 300), np.linspace(-1, 7, 300))
cs_map = model_pinn.sound_speed(10 ** lr, 10 ** lt)
fig, ax = plt.subplots(1, 2, figsize=(14, 5))
im = ax[0].pcolormesh(lr, lt, np.log10(cs_map), cmap='viridis', shading='auto')
plt.colorbar(im, ax=ax[0], label='log10 c_s, км/с')
ax[0].set(xlabel='log10 ρ, г/см³', ylabel='log10 T, K', title='PINN: скорость звука c_s (белое — c_s² ≤ 0)')
rho_line = np.logspace(-3, 4, 600)
for k, Tv in enumerate([10.0, 1e3, 1e5, 1e7]):
    ax[1].plot(rho_line, model_pinn.sound_speed(rho_line, np.full_like(rho_line, Tv)), color=f'C{k}', label=f'PINN, T={Tv:g} K')
ax[1].axhline(C_LIGHT_KMS, color='r', ls='--', label='скорость света c')
ax[1].set(xscale='log', yscale='log', xlabel='ρ, г/см³', ylabel='c_s, км/с', title='c_s(ρ): 0 < c_s < c')
ax[1].grid(alpha=0.3); ax[1].legend(fontsize=8)
plt.tight_layout()
plt.savefig(FIGURES / 'pinn_sound_speed.png', dpi=130, bbox_inches='tight')
plt.show()
