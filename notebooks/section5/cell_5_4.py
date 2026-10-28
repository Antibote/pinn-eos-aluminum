# 5.4 SVD-сжатие внутренних слоёв + дообучение с физическими штрафами («безопасный путь»)
# Внутренние слои 256→256 заменяются произведением матриц ранга r: W ≈ (U_r·S_r)·V_rᵀ (256→r→256).
# Первый слой 2→256 и последний 256→2 не сжимаются — они маленькие и критичны для физики.
# Обрезка ранга сразу портит точность (веса не низкоранговые), поэтому дальше — короткое дообучение
# с теми же физическими штрафами и исходной PINN в роли учителя.
RANKS = [64, 32]            # r = 64 — сжатие внутренних слоёв в 2 раза, r = 32 — в 4 раза
STEPS_FT = 1000             # 500–1000 шагов дообучения
LAM = {'mono_rho': 1, 'mono_T': 1, 'sound': 1}

if 'trainer' not in globals():   # trainer создаётся в 5.2; если 5.2 не запускали — создаём здесь
    def to_t(a):
        return torch.tensor(np.asarray(a), dtype=torch.float32, device=DEVICE)
    trainer = pn.Trainer(*(to_t(df[c].values) for c in ['rho', 'T', 'P', 'e']),
                         idx_train=torch.tensor(split['train'], device=DEVICE),
                         idx_val=torch.tensor(split['val'], device=DEVICE),
                         w_train=to_t(em.region_weights(df.iloc[split['train']])))
(ARTIFACTS / 'optimized').mkdir(exist_ok=True)

def test_metrics(net):
    met, _ = mt.evaluate(inf.EOSModel.from_net(net, device=DEVICE, dtype=torch.float64).predict, test_df, '')
    return {k: met[k] for k in ['err_P_median_%', 'err_P_p99_%', 'err_e_median_%', 'err_e_p99_%',
                                'viol_dP_drho_%', 'viol_de_dT_%', 'viol_cs_%']}

rows = [{'модель': 'PINN 256×4 (исходная)', 'параметров': sum(p.numel() for p in teacher.parameters()),
         'файл float16, МБ': inf.save_net(teacher, ARTIFACTS / 'optimized' / 'pinn_256x4_fp16.pt', fp16=True),
         **test_metrics(teacher),
         'CPU torch, с': bl.benchmark(inf.EOSModel.from_net(teacher, device='cpu'), n=100_000)}]
svd_nets = {}
for r in RANKS:
    print(f'=== SVD r = {r}: доля Σσ² в первых r сингулярных числах по слоям — {np.round(inf.svd_energy(teacher, r), 3)}')
    net = inf.compress_svd(teacher, r)
    for p in net.parameters():
        p.requires_grad_(True)
    rows.append({'модель': f'SVD r={r}, без дообучения', 'параметров': sum(p.numel() for p in net.parameters()),
                 **test_metrics(net)})
    hist = trainer.fit(net, lam=LAM, steps=STEPS_FT, n_coll=1024, lr=5e-4, ramp=0.0, teacher=teacher, eval_every=250)
    path = ARTIFACTS / 'optimized' / f'pinn_svd{r}_fp16.pt'
    row = {'модель': f'SVD r={r} + дообучение', 'параметров': sum(p.numel() for p in net.parameters()),
           'файл float16, МБ': inf.save_net(net, path, fp16=True), **test_metrics(net),
           'CPU torch, с': bl.benchmark(inf.EOSModel(path, NORM, device='cpu'), n=100_000),
           'дообучение, с': hist['time_s']}
    inf.export_onnx(net, ARTIFACTS / 'optimized' / f'pinn_svd{r}.onnx')
    row['CPU ONNX Runtime, с'] = bl.benchmark(inf.OnnxEOS(ARTIFACTS / 'optimized' / f'pinn_svd{r}.onnx'), n=100_000)
    if torch.cuda.is_available():
        row['GPU float32, с'] = bl.benchmark(inf.EOSModel(path, NORM, device='cuda'), n=100_000)
    rows.append(row)
    svd_nets[r] = net

svd_table = pd.DataFrame(rows).set_index('модель')
svd_table.to_csv(ARTIFACTS / 'metrics_svd.csv')
display(svd_table.T.round(4))

ref = svd_table.iloc[0]
for r in RANKS:
    row = svd_table.loc[f'SVD r={r} + дообучение']
    print(f"SVD r={r}: параметров ×{ref['параметров'] / row['параметров']:.1f} меньше, "
          f"файл {row['файл float16, МБ']:.3f} МБ (×{ref['файл float16, МБ'] / row['файл float16, МБ']:.1f}), "
          f"ошибка P {row['err_P_median_%']:.3f}% (было {ref['err_P_median_%']:.3f}%), "
          f"нарушения ∂P/∂ρ {row['viol_dP_drho_%']:.3f}%, ∂e/∂T {row['viol_de_dT_%']:.3f}%, c_s {row['viol_cs_%']:.3f}%")

# График: ошибка до и после дообучения, размер
names = list(svd_table.index)
fig, ax = plt.subplots(1, 2, figsize=(14, 4.5))
ax[0].bar(names, svd_table['err_P_median_%'], color='C0', label='P')
ax[0].bar(names, svd_table['err_e_median_%'], color='C3', alpha=0.6, label='e')
ax[0].set(yscale='log', ylabel='медианная ошибка на test, %', title='SVD: точность до и после дообучения')
sizes = svd_table['файл float16, МБ'].dropna()
ax[1].bar(sizes.index, sizes, color='C1')
ax[1].set(ylabel='МБ (веса float16)', title='Размер файла')
for a in ax:
    a.tick_params(axis='x', rotation=30); a.grid(axis='y', alpha=0.3)
ax[0].legend()
plt.tight_layout()
plt.savefig(FIGURES / 'svd_compression.png', dpi=130, bbox_inches='tight')
plt.show()
save_to_repo('5.4: SVD-сжатие внутренних слоёв + дообучение с физическими штрафами')
