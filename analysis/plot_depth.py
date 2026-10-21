"""Plots of the depth analyses: depth_profiles.csv (depth_profiles.py) and role_<ckpt>_{contributions,ablations}.csv
(paralesn_role.py). Usage: plot_depth.py RESULTS_DIR"""
import sys
from pathlib import Path

import matplotlib
import pandas as pd

matplotlib.use('Agg')
import matplotlib.pyplot as plt  # noqa: E402

d = Path(sys.argv[1])
prof = pd.read_csv(d / 'depth_profiles.csv')
blocks = prof[prof['layer'].str.startswith('block')]
steps = sorted(blocks['step'].unique())
show = [s for s in (25000, 68176, 170440, steps[-1]) if s in steps]
colors = dict(zip(show, ('#9DB4D8', '#6A8CC1', '#3D64A6', '#1B2F5B')))

fig, ax = plt.subplots(2, 2, figsize=(11, 7.5))
for s in show:
    b = blocks[blocks['step'] == s].sort_values('depth')
    lab = f'step {s // 1000}k'
    ax[0, 0].plot(b['depth'], b['gate1'], 'o-', color=colors[s], label=lab)
    ax[0, 1].plot(b['depth'], b['gate2'], 'o-', color=colors[s], label=lab)
    ax[1, 0].plot(b['depth'], b['mlp_w2_spec'], 'o-', color=colors[s], label=lab)
    ax[1, 1].plot(b['depth'], b['gain1'], 'o-', color=colors[s], label=lab)
ax[0, 0].set_title('AdaLN gate1 (ESN + conv branch), RMS')
ax[0, 1].set_title('AdaLN gate2 (MLP branch), RMS')
ax[1, 0].set_title('MLP output matrix W2, spectral norm')
ax[1, 1].set_title('gain of the ESN/conv input: |LN weight * (1 + scale1)|')
for a in ax.flat:
    a.set_xlabel('block (0 = first, 11 = last)')
    a.grid(alpha=0.3)
    a.legend(fontsize=8)
fig.suptitle('DiESN depth profiles (EMA weights)')
fig.tight_layout()
fig.savefig(d / 'depth_profiles.png', dpi=140)
print('saved', d / 'depth_profiles.png')

for contrib_path in sorted(d.glob('role_*_contributions.csv')):
    tag = contrib_path.name.replace('role_', '').replace('_contributions.csv', '')
    c = pd.read_csv(contrib_path)
    fig, ax = plt.subplots(1, 2, figsize=(12, 4.5), sharey=True)
    for a, p, title in ((ax[0], 'x0', 'x0 pass (clean context)'), (ax[1], 'xt', 'x_t pass, masked positions')):
        sub = c[c['pass'] == p]
        for metric, col in (('esn/resid', '#4C72B0'), ('conv/resid', '#DD8452'), ('mlp/resid', '#55A868')):
            m = sub[sub['metric'] == metric].copy()
            m['depth'] = m['layer'].str.replace('block', '').astype(int)
            m = m.sort_values('depth')
            a.plot(m['depth'], m['mean'], 'o-', color=col, label=metric.replace('/resid', ''))
        at = sub[(sub['layer'] == 'attention') & (sub['metric'] == 'attn/resid')]['mean']
        if len(at):
            a.plot([12], [at.iloc[0]], 's', color='#C44E52', ms=9, label='attention (after block 11)')
        a.set_title(title)
        a.set_xlabel('block')
        a.grid(alpha=0.3)
    ax[0].set_ylabel('|update written| / |residual stream entering|')
    ax[1].legend(fontsize=8)
    fig.suptitle(f'Relative size of the updates of each branch ({tag})')
    fig.tight_layout()
    fig.savefig(d / f'contributions_{tag}.png', dpi=140)
    print('saved', d / f'contributions_{tag}.png')

for abl_path in sorted(d.glob('role_*_ablations.csv')):
    tag = abl_path.name.replace('role_', '').replace('_ablations.csv', '')
    a = pd.read_csv(abl_path)
    per = a[a['block'] != 'all'].copy()
    if per.empty:
        continue
    per['block'] = per['block'].astype(int)
    fig, ax = plt.subplots(figsize=(8, 4.2))
    for name, col in (('no esn', '#4C72B0'), ('no conv', '#DD8452'), ('no mlp', '#55A868')):
        s = per[per['ablation'] == name].sort_values('block')
        ax.plot(s['block'], s['delta'], 'o-', color=col, label=name.replace('no ', 'without '))
    ax.set_yscale('log')
    ax.set_xlabel('block whose branch is removed')
    ax.set_ylabel('increase of the NELBO (nats per token)')
    ax.set_title(f'Removing one branch in one block ({tag})')
    ax.grid(alpha=0.3, which='both')
    ax.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(d / f'ablations_per_block_{tag}.png', dpi=140)
    print('saved', d / f'ablations_per_block_{tag}.png')
