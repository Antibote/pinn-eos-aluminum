# 5.3 Измеримый эффект оптимизаций: размер, скорость (CPU / GPU), точность и физика на test
(ARTIFACTS / 'optimized').mkdir(exist_ok=True)
nets = {'PINN 256×4': teacher, 'PINN 128×4 (дистилл.)': students['pinn_128x4'],
        'PINN 64×4 (дистилл.)': students['pinn_64x4']}
rows = []
for name, net in nets.items():
    tag = name.split()[1].replace('×', 'x')
    fp16_path = ARTIFACTS / 'optimized' / f'pinn_{tag}_fp16.pt'
    onnx_path = ARTIFACTS / 'optimized' / f'pinn_{tag}.onnx'
    row = {'модель': name, 'параметров': sum(p.numel() for p in net.parameters()),
           'файл float32, МБ': inf.save_net(net, ARTIFACTS / 'optimized' / f'pinn_{tag}.pt'),
           'файл float16, МБ': inf.save_net(net, fp16_path, fp16=True)}
    # точность и нарушения — у float16-версии (её и отдаём), считаем во float64
    model16 = inf.EOSModel(fp16_path, NORM, device=DEVICE, dtype=torch.float64)
    met, _ = mt.evaluate(model16.predict, test_df, name)
    row.update({k: met[k] for k in ['err_P_median_%', 'err_P_p99_%', 'err_e_median_%', 'err_e_p99_%',
                                    'viol_dP_drho_%', 'viol_de_dT_%', 'viol_cs_%']})
    # скорость на 10^5 запросов
    row['CPU torch, с'] = bl.benchmark(inf.EOSModel(fp16_path, NORM, device='cpu'), n=100_000)
    inf.export_onnx(net, onnx_path)
    row['CPU ONNX Runtime, с'] = bl.benchmark(inf.OnnxEOS(onnx_path), n=100_000)
    if torch.cuda.is_available():
        row['GPU float32, с'] = bl.benchmark(inf.EOSModel(fp16_path, NORM, device='cuda'), n=100_000)
        fast = inf.GpuFp16EOS(net)
        row['GPU float16, с'] = bl.benchmark(fast, n=100_000)
        m16, _ = mt.evaluate(fast.predict, test_df.iloc[::10], name)   # точность при счёте во float16
        row['GPU float16: ошибка P, медиана %'] = m16['err_P_median_%']
    rows.append(row)

with open(ARTIFACTS / 'metrics_baseline.json') as f:
    base = json.load(f)
rows.insert(0, {'модель': 'Бейзлайн (таблица)', 'файл float32, МБ': base['size_MB'],
                'err_P_median_%': base['err_P_median_%'], 'err_P_p99_%': base['err_P_p99_%'],
                'err_e_median_%': base['err_e_median_%'], 'err_e_p99_%': base['err_e_p99_%'],
                'viol_dP_drho_%': base['viol_dP_drho_%'], 'viol_de_dT_%': base['viol_de_dT_%'],
                'CPU torch, с': base['time_1e5_s']})
opt = pd.DataFrame(rows).set_index('модель')
opt.to_csv(ARTIFACTS / 'metrics_optimization.csv')
display(opt.T.round(4))

# Эффект относительно исходной PINN 256×4 float32 на CPU
ref = opt.loc['PINN 256×4']
best_cpu = opt[['CPU torch, с', 'CPU ONNX Runtime, с']].min(axis=1)
print('Эффект оптимизации (относительно PINN 256×4, float32, CPU torch):')
for name in nets:
    print(f"  {name:24s} размер ×{ref['файл float32, МБ'] / opt.loc[name, 'файл float16, МБ']:.1f} меньше, "
          f"CPU ×{ref['CPU torch, с'] / best_cpu[name]:.1f} быстрее, "
          f"ошибка P {opt.loc[name, 'err_P_median_%']:.3f}% (было {ref['err_P_median_%']:.3f}%)")

# График: размер, время на CPU и ошибка
names = list(opt.index)
fig, ax = plt.subplots(1, 3, figsize=(16, 4.5))
ax[0].bar(names, opt['файл float32, МБ'], color='0.7', label='float32')
ax[0].bar(names, opt['файл float16, МБ'].fillna(opt['файл float32, МБ']), color='C0', label='float16')
ax[0].axhline(10, color='r', ls='--', label='лимит 10 МБ')
ax[0].set(yscale='log', ylabel='МБ', title='Размер модели')
ax[1].bar(names, opt['CPU torch, с'], color='0.7', label='PyTorch')
ax[1].bar(names, best_cpu.fillna(opt['CPU torch, с']), color='C1', label='лучшее (PyTorch / ONNX)')
ax[1].set(yscale='log', ylabel='с на 10^5 запросов', title='Скорость на CPU')
ax[2].bar(names, opt['err_P_median_%'], color='C2', label='P')
ax[2].bar(names, opt['err_e_median_%'], color='C3', alpha=0.6, label='e')
ax[2].set(yscale='log', ylabel='медианная ошибка, %', title='Точность (test)')
for a in ax:
    a.tick_params(axis='x', rotation=30); a.grid(axis='y', alpha=0.3); a.legend(fontsize=8)
plt.tight_layout()
plt.savefig(FIGURES / 'optimization.png', dpi=130, bbox_inches='tight')
plt.show()
save_to_repo('5.3: оптимизация — размер, скорость, точность')
