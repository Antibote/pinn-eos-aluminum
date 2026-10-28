# 3.2 Обучение PINN (прогресс печатается каждые 1000 шагов)
torch.manual_seed(0)
pinn = em.EOSNet(norm, hidden=CFG['hidden'], layers=CFG['layers'], activation=CFG['activation']).to(DEVICE)
hist = trainer.fit(pinn, lam=CFG['lam'], steps=STEPS, loss=CFG['loss'], grad=CFG['grad'],
                   batch=CFG['batch'], n_coll=CFG['n_coll'], lr=CFG['lr'])
print(f"Обучение: {hist['time_s'] / 60:.1f} мин, лучший val MSE {hist['best_val']:.3e}")
torch.save({'state_dict': pinn.state_dict(), 'config': pinn.config, 'train': CFG, 'steps': STEPS},
           ARTIFACTS / 'pinn.pt')

fig, ax = plt.subplots(1, 2, figsize=(14, 4))
ax[0].plot(hist['step'], hist['val']); ax[0].set(yscale='log', xlabel='шаг', ylabel='MSE по данным (val)', title='Точность')
ax[1].plot(hist['step'], hist['phys']); ax[1].set(yscale='symlog', xlabel='шаг', ylabel='λ · штраф (train)', title='Физика')
for a in ax:
    a.grid(alpha=0.3)
plt.tight_layout()
plt.savefig(FIGURES / 'pinn_learning_curve.png', dpi=130, bbox_inches='tight')
plt.show()
save_to_repo('3.2: PINN (GELU, MSE, штрафы ∂P/∂ρ, ∂e/∂T, c_s) — веса и кривая обучения')
