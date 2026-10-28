# 5.5 Инференс на C: прямой проход обученной PINN без Python и PyTorch (код — csrc/eos_pinn.c, src/eos/cinfer.py)
# Обучение и autograd остаются в PyTorch; на C — только нормировка → слои Linear + GELU → обратная нормировка.
# Зачем: уравнение состояния вызывают из расчётных кодов (C/Fortran) миллионы раз по одной точке на ячейку —
# там важна задержка одного вызова и отсутствие зависимостей, а не пакет из 10^5 точек на GPU.
import importlib, json, subprocess, sys, time
import numpy as np, pandas as pd, torch
import eos.normalize as nz, eos.model as em, eos.metrics as mt, eos.baseline as bl, eos.inference as inf, eos.cinfer as ci
for m_ in (nz, em, mt, bl, inf, ci):
    importlib.reload(m_)
try:
    import onnxruntime
except ImportError:
    subprocess.run([sys.executable, '-m', 'pip', 'install', '-q', 'onnx', 'onnxruntime'], check=True)
NORM = ARTIFACTS / 'normalization.json'                       # всё нужное — здесь: ячейку можно запускать сразу после раздела 0
test_df = df.iloc[np.load(ARTIFACTS / 'split.npz')['test']]

C_DIR = ARTIFACTS / 'c'
C_DIR.mkdir(exist_ok=True)
models_c = {'PINN 256×4': ARTIFACTS / 'pinn.pt', 'PINN SVD r=64': ARTIFACTS / 'optimized' / 'pinn_svd64_fp16.pt'}
lib = ci.build()
exe = ci.CSRC / 'build' / 'example'
subprocess.run(['gcc', '-O3', '-march=native', '-ffast-math', '-fopenmp', str(ci.CSRC / 'example.c'),
                str(ci.CSRC / 'eos_pinn.c'), '-o', str(exe), '-lm'], check=True)

def latency(f, n=2000):
    """Средняя задержка одного запроса, мкс."""
    pts = test_df[['rho', 'T']].to_numpy()[:n]
    f(pts[0, :1], pts[0, 1:])
    t = time.perf_counter()
    for r, T_ in pts:
        f(np.array([r]), np.array([T_]))
    return (time.perf_counter() - t) / n * 1e6

rows = []
for name, path in models_c.items():
    net = inf.load_eosnet(path, NORM)
    bin_path = C_DIR / (path.stem + '.bin')
    size = ci.export_weights(net, bin_path)
    cm = ci.CEOS(bin_path)
    tm = inf.EOSModel.from_net(net, device='cpu')
    onnx_path = C_DIR / (path.stem + '.onnx'); inf.export_onnx(net, onnx_path); om = inf.OnnxEOS(onnx_path)
    # точность C относительно PyTorch и на test
    P_t, e_t = inf.EOSModel.from_net(net, device='cpu', dtype=torch.float64).predict(test_df['rho'].values, test_df['T'].values)
    P_c, e_c = cm.predict(test_df['rho'].values, test_df['T'].values)
    met, _ = mt.evaluate(cm.predict, test_df, name)
    # чистый C (без Python): задержка одного вызова и пакет 10^5
    out = subprocess.run([str(exe), str(bin_path)], capture_output=True, text=True, check=True).stdout
    c_one = float(out.split('eos_predict_one:')[1].split()[0]); c_batch = float(out.split('10^5 точек:')[1].split()[0])
    rows.append({'модель': name, 'файл весов C, МБ': size,
                 'макс. отличие P от PyTorch': float(np.max(np.abs(P_c / P_t - 1))),
                 'ошибка P на test, медиана %': met['err_P_median_%'], 'ошибка e на test, медиана %': met['err_e_median_%'],
                 '10^5 запросов: PyTorch CPU, с': bl.benchmark(tm, n=100_000),
                 '10^5 запросов: ONNX Runtime, с': bl.benchmark(om, n=100_000),
                 '10^5 запросов: C (из Python), с': bl.benchmark(cm, n=100_000),
                 '10^5 запросов: чистый C, с': c_batch,
                 '1 запрос: PyTorch, мкс': latency(tm.predict),
                 '1 запрос: ONNX Runtime, мкс': latency(om.predict),
                 '1 запрос: C (из Python), мкс': latency(lambda r, T_: cm.predict_one(r[0], T_[0])),
                 '1 запрос: чистый C, мкс': c_one})
c_table = pd.DataFrame(rows).set_index('модель')
c_table.to_csv(ARTIFACTS / 'metrics_c.csv')
display(c_table.T.round(5))
for name in c_table.index:
    r = c_table.loc[name]
    print(f"{name}: один запрос — чистый C {r['1 запрос: чистый C, мкс']:.1f} мкс против PyTorch {r['1 запрос: PyTorch, мкс']:.0f} мкс "
          f"(×{r['1 запрос: PyTorch, мкс'] / r['1 запрос: чистый C, мкс']:.0f}) и ONNX {r['1 запрос: ONNX Runtime, мкс']:.0f} мкс "
          f"(×{r['1 запрос: ONNX Runtime, мкс'] / r['1 запрос: чистый C, мкс']:.1f}); точность совпадает с PyTorch до "
          f"{r['макс. отличие P от PyTorch']:.0e}")
print('\nВывод программы на чистом C (csrc/example.c):\n' + subprocess.run([str(exe), str(C_DIR / 'pinn.bin')], capture_output=True, text=True).stdout)
save_to_repo('5.5: инференс PINN на C — точность и скорость')
