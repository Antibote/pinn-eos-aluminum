# 5.1 Исходная точка: PINN 256×4 из раздела 3 — размер, скорость, точность
import importlib, json, subprocess, sys, torch
import eos.normalize as nz, eos.model as em, eos.pinn as pn, eos.metrics as mt, eos.baseline as bl, eos.inference as inf
for m in (nz, em, mt, pn, bl, inf):
    importlib.reload(m)
try:
    import onnx, onnxruntime
except ImportError:
    subprocess.run([sys.executable, '-m', 'pip', 'install', '-q', 'onnx', 'onnxruntime'], check=True)
pd.set_option('display.width', 250)

NORM = ARTIFACTS / 'normalization.json'
split = dict(np.load(ARTIFACTS / 'split.npz'))
test_df = df.iloc[split['test']]
teacher = inf.load_eosnet(ARTIFACTS / 'pinn.pt', NORM, device=DEVICE)
for p in teacher.parameters():
    p.requires_grad_(False)

def flops_per_query(net):
    """Умножений-сложений на один запрос (по слоям Linear) — от этого зависит время на CPU."""
    return sum(2 * m.in_features * m.out_features for m in net.modules() if isinstance(m, torch.nn.Linear))

with open(ARTIFACTS / 'metrics_pinn.json') as f:
    start = json.load(f)
cpu_start = bl.benchmark(inf.EOSModel(ARTIFACTS / 'pinn.pt', NORM, device='cpu'), n=100_000)
print(f"PINN {teacher.config}: файл {(ARTIFACTS / 'pinn.pt').stat().st_size / 2**20:.3f} МБ, "
      f"{flops_per_query(teacher) / 1e3:.0f} тыс. операций на запрос")
print(f"Время на 10^5 запросов: CPU {cpu_start:.3f} с, GPU {start['time_1e5_s']:.4f} с; "
      f"ошибка P {start['err_P_median_%']:.3f}%, e {start['err_e_median_%']:.3f}% (медианы)")
print('Почти всё время на CPU — умножение матриц 256×256: его и уменьшаем (5.2), плюс ускоряем сам расчёт (5.3)')
