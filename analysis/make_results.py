"""Results of the run diesn_smdm_bs16 from its full history (local wandb files, read_wandb_local.py): validation and
training CSVs, per-epoch means, a power-law fit of val/nll and the plot of the validation curve.
Usage: make_results.py HISTORY.pkl OUT_DIR"""
import sys
from pathlib import Path

import matplotlib
import numpy as np
import pandas as pd
from scipy.optimize import curve_fit

matplotlib.use('Agg')
import matplotlib.pyplot as plt  # noqa: E402

STEPS_PER_EPOCH = 17044
PAPER = {'BD3-LM (DiT) L\'=16': 3.103, 'BD3-LM (DiT) L\'=4': 3.032}  # ln(22.27), ln(20.73): 60 epochs (1M steps)

df = pd.read_pickle(sys.argv[1])
out = Path(sys.argv[2])
out.mkdir(parents=True, exist_ok=True)

val = df[df['val/nll'].notna()][['trainer/global_step', 'epoch', 'val/nll', 'val/ppl', 'val/bpd']].copy()
eps = df[df['sampling_eps_min'].notna()][['trainer/global_step', 'sampling_eps_min', 'sampling_eps_max']]
val = val.merge(eps, on='trainer/global_step', how='left').rename(columns={'trainer/global_step': 'step'})
val['epochs_done'] = val['step'] / STEPS_PER_EPOCH
val.round(5).to_csv(out / 'validation.csv', index=False)

train_cols = ['trainer/global_step', 'epoch', 'trainer/loss', 'trainer/time_h', 'transparency/scale',
              'transparency/centre', 'transparency/steepness', 'trainer/lr/pg1', 'trainer/lr/pg2']
tr = df[df['trainer/loss'].notna()][[c for c in train_cols if c in df.columns]]
tr = tr.groupby('trainer/global_step', as_index=False).last().rename(columns={'trainer/global_step': 'step'})
tr.round(6).to_csv(out / 'train.csv', index=False)

val['epoch_idx'] = (val['step'] // STEPS_PER_EPOCH).astype(int)
ep = val.groupby('epoch_idx').agg(val_nll_mean=('val/nll', 'mean'), val_nll_min=('val/nll', 'min'),
                                  n_validations=('val/nll', 'size')).reset_index()
ep['ppl_of_mean'] = np.exp(ep['val_nll_mean'])
ep['delta_vs_previous_epoch'] = ep['val_nll_mean'].diff()
ep.round(4).to_csv(out / 'epochs.csv', index=False)
print(ep.round(4).to_string(index=False))

# power law L(s) = L_inf + A (s / 1000)^-alpha on the validations from 30k steps
f = lambda s, linf, a, alpha: linf + a * (s / 1000) ** (-alpha)  # noqa: E731
sel = val['step'] >= 30000
p, cov = curve_fit(f, val.loc[sel, 'step'], val.loc[sel, 'val/nll'], p0=[3.3, 2.0, 0.4], maxfev=50000)
err = np.sqrt(np.diag(cov))
print(f'fit: L_inf {p[0]:.3f} +- {err[0]:.3f}, A {p[1]:.3f}, alpha {p[2]:.3f}')
for e in (25, 50, 60):
    print(f'  {e} epochs ({e * STEPS_PER_EPOCH:,} steps): {f(e * STEPS_PER_EPOCH, *p):.3f}')

fig, ax = plt.subplots(figsize=(9, 5))
ax.plot(val['epochs_done'], val['val/nll'], 'o', ms=3, color='#4C72B0', alpha=0.6, label='val/nll (each validation)')
ax.plot(ep['epoch_idx'] + 0.5, ep['val_nll_mean'], '-', lw=2, color='#4C72B0', label='val/nll, mean per epoch')
xs = np.linspace(30000, 60 * STEPS_PER_EPOCH, 300)
ax.plot(xs / STEPS_PER_EPOCH, f(xs, *p), '--', color='gray', lw=1,
        label=f'power-law fit (asymptote {p[0]:.2f})')
for (name, v), c in zip(PAPER.items(), ('#C44E52', '#DD8452')):
    ax.axhline(v, color=c, ls=':', lw=1.5, label=f'{name}: {v:.2f} (paper, 60 epochs)')
ax.axvline(25000 / STEPS_PER_EPOCH, color='k', lw=0.8, alpha=0.4)
ax.text(25000 / STEPS_PER_EPOCH + 0.3, 3.95, 'resume by epochs\n(step 25k)', fontsize=8, alpha=0.7)
ax.set_xlabel('epochs of OpenWebText (17,044 steps of 512 x 1024 tokens each)')
ax.set_ylabel('val/nll (NELBO per token, EMA weights)')
ax.set_title('BD3-LM + DiESN + SMDM soft masking, block size 16 (run diesn_smdm_bs16)')
ax.set_xlim(0, 60)
ax.set_ylim(2.95, 4.2)
ax.grid(alpha=0.3)
ax.legend(fontsize=8, loc='upper right')
fig.tight_layout()
fig.savefig(out / 'val_curve.png', dpi=150)
print('saved', out / 'val_curve.png')
