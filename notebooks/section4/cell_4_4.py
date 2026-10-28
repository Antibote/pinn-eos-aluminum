# 4.4 Сжатие: float16, int8 (динамическая квантизация), int8 только весов. Точность — на общем test
test_df = df.iloc[np.load(ARTIFACTS / 'split.npz')['test']]

FP16_PATH, INT8W_PATH = ARTIFACTS / 'pinn_fp16.pt', ARTIFACTS / 'pinn_int8w.pt'
inf.save_fp16(model.net, FP16_PATH)
net_w8, packed = inf.compress_int8_weights(model.net)
inf.save_int8_weights(net_w8, packed, INT8W_PATH)
net_q8 = inf.compress_int8(model.net)
q8_buf = ARTIFACTS / 'pinn_int8_dynamic.pt'
torch.save(net_q8.state_dict(), q8_buf)

variants = {
    'PINN float32 (исходная)': (model64, MODEL),
    'PINN float16': (inf.EOSModel(FP16_PATH, NORM, device=DEVICE, dtype=torch.float64), FP16_PATH),
    'PINN int8 (только веса)': (inf.EOSModel.from_net(inf.load_int8_weights(INT8W_PATH, NORM), device=DEVICE,
                                                     dtype=torch.float64), INT8W_PATH),
    'PINN int8 (динамическая, CPU)': (inf.EOSModel.from_net(net_q8), q8_buf),
}
rows = []
for name, (mm, path) in variants.items():
    met, _ = mt.evaluate(mm.predict, test_df, name)
    met['size_MB'] = path.stat().st_size / 2**20
    rows.append(met)
compress = pd.DataFrame(rows).set_index('model')
compress.to_csv(ARTIFACTS / 'metrics_compression.csv')
display(compress[['size_MB', 'err_P_median_%', 'err_P_p99_%', 'err_e_median_%', 'err_e_p99_%',
                  'viol_dP_drho_%', 'viol_de_dT_%']].round(4))
q8_buf.unlink()   # динамическая int8 непригодна (см. таблицу) — в репозиторий не кладём
