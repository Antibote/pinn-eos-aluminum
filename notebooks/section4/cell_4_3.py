# 4.3 Экспорт: TorchScript и ONNX (нормировка зашита внутрь — вход ρ, T, выход P, e)
import subprocess, sys
try:
    import onnx, onnxruntime
except ImportError:
    subprocess.run([sys.executable, '-m', 'pip', 'install', '-q', 'onnx', 'onnxruntime'], check=True)

TS_PATH, ONNX_PATH = ARTIFACTS / 'pinn_torchscript.pt', ARTIFACTS / 'pinn.onnx'
ts = inf.export_torchscript(model.net, TS_PATH)
inf.export_onnx(model.net, ONNX_PATH)
onnx_model = inf.OnnxEOS(ONNX_PATH)

# Совпадают ли ответы экспортированных моделей с исходной (float64)
P0, e0 = model64.predict(rho_q, T_q)
with torch.no_grad():
    P_ts, e_ts = (x.numpy() for x in ts(torch.tensor(rho_q, dtype=torch.float32),
                                         torch.tensor(T_q, dtype=torch.float32)))
P_ox, e_ox = onnx_model.predict(rho_q, T_q)
rel = lambda P, e: (np.max(np.abs(P / P0 - 1)), np.max(np.abs(e - e0) / np.maximum(np.abs(e0), 1)))
export = pd.DataFrame({
    'TorchScript': [TS_PATH.stat().st_size / 2**20, *rel(P_ts, e_ts), np.nan],
    'ONNX': [ONNX_PATH.stat().st_size / 2**20, *rel(P_ox, e_ox), bl.benchmark(onnx_model, n=100_000)],
}, index=['размер, МБ', 'макс. отн. отличие P', 'макс. отличие e (как в метрике)', 'время 10^5 (ONNX Runtime, CPU), с'])
display(export.T)
print('Загрузка без нашего кода: torch.jit.load(path)(rho, T)  или  onnxruntime.InferenceSession(path)')
