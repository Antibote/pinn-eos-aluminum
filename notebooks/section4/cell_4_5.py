# 4.5 Единая таблица сравнения baseline / Vanilla / PINN (+ сжатая PINN) — на одном test
rows = []
for name in ['metrics_baseline.json', 'metrics_vanilla.json']:
    with open(ARTIFACTS / name) as f:
        rows.append(json.load(f))
pinn_met, _ = mt.evaluate(model64.predict, test_df, 'PINN')
pinn_met['size_MB'] = MODEL.stat().st_size / 2**20
fp16_met = compress.loc['PINN float16'].to_dict() | {'model': 'PINN float16'}
gpu = 'PINN cuda float32' if 'PINN cuda float32' in speed else 'PINN cpu float32'
pinn_met['time_1e5_s'] = fp16_met['time_1e5_s'] = speed[gpu]
pinn_met['time_cpu_1e5_s'] = fp16_met['time_cpu_1e5_s'] = speed['PINN cpu float32']
rows += [pinn_met, fp16_met]

final = pd.DataFrame(rows).set_index('model')
base_row = final.index.str.startswith('Baseline')            # бейзлайн считается только на CPU
final.loc[base_row, 'time_cpu_1e5_s'] = final.loc[base_row, 'time_1e5_s']
cols = {'err_P_median_%': 'ошибка P, медиана %', 'err_P_p99_%': 'ошибка P, p99 %',
        'err_e_median_%': 'ошибка e, медиана %', 'err_e_p99_%': 'ошибка e, p99 %',
        'viol_dP_drho_%': '∂P/∂ρ ≤ 0, %', 'viol_de_dT_%': '∂e/∂T < 0, %',
        'size_MB': 'размер, МБ', 'time_1e5_s': 'время 10^5, с (сети — GPU, бейзлайн — CPU)',
        'time_cpu_1e5_s': 'время 10^5 на CPU, с'}
table = final[list(cols)].rename(columns=cols)
table.to_csv(ARTIFACTS / 'metrics_final.csv')
try:
    md = table.to_markdown(floatfmt='.4g')            # таблица для README и отчёта
    (ARTIFACTS / 'metrics_final.md').write_text(md)
except ImportError:
    print('Для metrics_final.md нужен пакет tabulate: pip install tabulate')
display(table.T.round(4))
print('Нарушения у Vanilla посчитаны во float32 (раздел 2) — часть из них шум округления.')
save_to_repo('4: EOSModel, скорость, экспорт TorchScript/ONNX, сжатие, итоговая таблица')
