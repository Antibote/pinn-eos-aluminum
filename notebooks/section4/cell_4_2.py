# 4.2 Размер модели и время на 10^5 запросов (CPU и GPU)
size = {'файл pinn.pt, МБ': MODEL.stat().st_size / 2**20,
        'параметров': sum(p.numel() for p in model.net.parameters())}
print(pd.Series(size).to_string(), '\nЛимит по заданию: 10 МБ')

speed = {}
for dev in ['cpu'] + (['cuda'] if torch.cuda.is_available() else []):
    for dt in (torch.float32, torch.float64):
        mm = inf.EOSModel(MODEL, NORM, device=dev, dtype=dt)
        speed[f'PINN {dev} {str(dt)[6:]}'] = bl.benchmark(mm, n=100_000)   # лучшее из 5 прогонов
baseline = bl.BilinearEOS(df)
speed['Бейзлайн (билинейная, CPU)'] = bl.benchmark(baseline, n=100_000)
speed = pd.Series(speed, name='время на 10^5 запросов, с')
display(speed.round(4).to_frame())
with open(ARTIFACTS / 'metrics_speed.json', 'w') as f:
    json.dump({**size, **speed.to_dict()}, f, indent=2, ensure_ascii=False)
