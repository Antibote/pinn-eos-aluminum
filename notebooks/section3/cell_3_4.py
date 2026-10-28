# 3.4 Сравнение с базовой моделью: точность и нарушения физики на одном test (функция mt.evaluate)


class TorchEOS:
    """Обёртка для замера скорости: float32 на DEVICE, как в рабочем инференсе."""

    def __init__(self, model):
        self.model = model.eval()

    @torch.no_grad()
    def predict(self, rho, T):
        P, e = self.model(to_t(rho), to_t(T))
        return P.cpu().numpy(), e.cpu().numpy()


metrics, per_point = mt.evaluate(pn.numpy_predict(pinn), test_df, 'PINN')
metrics['time_1e5_s'] = bl.benchmark(TorchEOS(pinn), n=100_000)
metrics['size_MB'] = sum(p.numel() for p in pinn.parameters()) * 4 / 2**20
metrics.update({f'autograd_{k}': v for k, v in pn.physics_report(pinn, n=100_000).items()})
with open(ARTIFACTS / 'metrics_pinn.json', 'w') as f:
    json.dump(metrics, f, indent=2, ensure_ascii=False)

# Бейзлайн пересчитываем (быстро), чтобы в его метриках было и нарушение скорости звука
with open(ARTIFACTS / 'metrics_baseline.json') as f:
    base_old = json.load(f)
base_met, _ = mt.evaluate(bl.BilinearEOS(df, decimate=True).predict, test_df, 'Baseline (билинейная)')
base_met.update({k: base_old[k] for k in ('time_1e5_s', 'size_MB')})
rows = [base_met]
if (ARTIFACTS / 'metrics_vanilla.json').exists():          # Vanilla — из раздела 2 (без viol_cs_%)
    with open(ARTIFACTS / 'metrics_vanilla.json') as f:
        rows.append(json.load(f))
rows.append(metrics)
keys = ['err_P_median_%', 'err_P_p99_%', 'err_P_max_%', 'err_e_median_%', 'err_e_p99_%', 'err_e_max_%',
        'viol_dP_drho_%', 'viol_de_dT_%', 'viol_cs_%', 'viol_dP_dT_%', 'time_1e5_s', 'size_MB']
compare = pd.DataFrame(rows).set_index('model')
compare.to_csv(ARTIFACTS / 'metrics_compare.csv')
display(compare[keys].T.round(4))
print('Физика PINN через autograd (100 000 случайных точек):')
print(pd.Series({k: v for k, v in metrics.items() if k.startswith('autograd_')}).round(4).to_string())

# Карта ошибок на (ρ, T)
fig, ax = plt.subplots(1, 2, figsize=(13, 4.5))
for a, col, title in [(ax[0], 'err_P', 'ошибка P, %'), (ax[1], 'err_e', 'ошибка e, %')]:
    sc = a.scatter(per_point['rho'], per_point['T'], c=np.log10(per_point[col] * 100 + 1e-6), s=0.3,
                   cmap='viridis', vmin=-3, vmax=2, rasterized=True)
    a.set(xscale='log', yscale='log', xlabel='ρ, г/см³', ylabel='T, K', title=f'PINN: {title}')
    plt.colorbar(sc, ax=a, label='log10(ошибка, %)')
plt.tight_layout()
plt.savefig(FIGURES / 'pinn_errors.png', dpi=130, bbox_inches='tight')
plt.show()


def section(col, v):
    """Точки таблицы при col ≈ v: в каждой области свой ближайший узел (сетки разные)."""
    parts = []
    for _, g in df.groupby('region'):
        u = np.unique(g[col])
        k = np.abs(np.log(u / v)).argmin()
        if abs(np.log(u[k] / v)) < 0.02:
            parts.append(g[g[col] == u[k]])
    return pd.concat(parts)


# Сечения: таблица (точки) и PINN (линии); внизу — знак производных
predict = pn.numpy_predict(pinn)
fig, ax = plt.subplots(2, 2, figsize=(14, 9))
T_line, rho_line = np.logspace(-1, 7, 600), np.logspace(-3, 4, 600)
for k, rv in enumerate([0.01, 2.7, 100.0]):
    pts, r_ = section('rho', rv), np.full_like(T_line, rv)
    ax[0, 0].scatter(pts['T'], pts['e'], s=3, color=f'C{k}', alpha=0.3)
    ax[0, 0].plot(T_line, predict(r_, T_line)[1], color=f'C{k}', lw=1, label=f'ρ={rv:g}')
    d = pn.log_derivatives(pinn, to_t(r_), to_t(T_line), create_graph=False)
    ax[1, 0].plot(T_line, d['e_t'].detach().cpu().numpy(), color=f'C{k}', lw=1, label=f'ρ={rv:g}')
for k, Tv in enumerate([10.0, 1e3, 1e5]):
    pts, T_ = section('T', Tv), np.full_like(rho_line, Tv)
    ax[0, 1].scatter(pts['rho'], pts['P'], s=3, color=f'C{k}', alpha=0.3)
    ax[0, 1].plot(rho_line, predict(rho_line, T_)[0], color=f'C{k}', lw=1, label=f'T={Tv:g} K')
    d = pn.log_derivatives(pinn, to_t(rho_line), to_t(T_), create_graph=False)
    ax[1, 1].plot(rho_line, d['a'].detach().cpu().numpy(), color=f'C{k}', lw=1, label=f'T={Tv:g} K')
ax[0, 0].set(xscale='log', yscale='symlog', xlabel='T, K', ylabel='e, кДж/г', title='e(T): точки — таблица, линии — PINN')
ax[0, 1].set(xscale='log', yscale='log', xlabel='ρ, г/см³', ylabel='P, ГПа', title='P(ρ): точки — таблица, линии — PINN')
ax[1, 0].set(xscale='log', yscale='symlog', xlabel='T, K', ylabel='∂e/∂lnT', title='∂e/∂lnT ≥ 0 — термическая устойчивость')
ax[1, 1].set(xscale='log', xlabel='ρ, г/см³', ylabel='∂lnP/∂lnρ', title='∂lnP/∂lnρ > 0 — механическая устойчивость')
for a in ax.ravel():
    a.grid(alpha=0.3); a.legend(fontsize=8)
ax[1, 0].axhline(0, color='k', lw=0.8); ax[1, 1].axhline(0, color='k', lw=0.8)
plt.tight_layout()
plt.savefig(FIGURES / 'pinn_sections.png', dpi=130, bbox_inches='tight')
plt.show()

save_to_repo('3.3–3.4: PINN — скорость звука, метрики на test, карта ошибок, сечения')
