# 5.2 Дистилляция: маленькие сети учатся у PINN 256×4 (учитель) + те же физические штрафы.
# Учитель даёт ответы в любых точках области, поэтому ученик учится и между узлами таблицы
STUDENTS = {'pinn_128x4': (128, 4), 'pinn_64x4': (64, 4)}
STEPS_KD = 30_000                       # для пробного запуска — 3000
LAM = {'mono_rho': 1, 'mono_T': 1, 'sound': 1}

def to_t(a):
    return torch.tensor(np.asarray(a), dtype=torch.float32, device=DEVICE)

trainer = pn.Trainer(*(to_t(df[c].values) for c in ['rho', 'T', 'P', 'e']),
                     idx_train=torch.tensor(split['train'], device=DEVICE),
                     idx_val=torch.tensor(split['val'], device=DEVICE),
                     w_train=to_t(em.region_weights(df.iloc[split['train']])))
students = {}
for name, (hidden, layers) in STUDENTS.items():
    print(f'=== {name}: {hidden}×{layers}')
    torch.manual_seed(0)
    net = em.EOSNet(teacher.norm, hidden=hidden, layers=layers, activation='gelu').to(DEVICE)
    hist = trainer.fit(net, lam=LAM, steps=STEPS_KD, n_coll=1024, teacher=teacher)
    inf.save_net(net, ARTIFACTS / f'{name}.pt')
    students[name] = net
    print(f"    обучение {hist['time_s'] / 60:.1f} мин")
save_to_repo('5.2: дистилляция PINN в сети 128×4 и 64×4')
