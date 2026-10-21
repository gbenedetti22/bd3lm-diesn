"""Direction of the weights during training (CPU). Three points of the trajectory: W0 (initialization, rebuilt with
the seed of train.py), W_ema (EMA weights; with the num_updates warm-up of models/ema.py the decay is
(1+n)/(10+n) < 0.9999 for n < 90k, so the EMA at n = 25000 weighs step k ~ k^8: mean step 22.5k, std 2.3k) and
W (raw weights at step 25000), plus Adam's moments (m: gradient averaged over ~10 steps, v: over ~1000 steps).

Per tensor:
  cos_dir   cos(W - W_ema, W_ema - W0): is the recent motion in the direction of the whole trajectory?
  rate      speed of the recent motion along that direction / average speed since init (1 = same speed)
  cos_adam  cos(-m/sqrt(v), W - W_ema): does the current update continue the recent motion?
  snr       mean |m_hat| / sqrt(v_hat): 0.23 for pure noise (beta1 = 0.9), 1 for a constant gradient
  radial    fraction of the recent motion along W itself (norm growth)
"""
import sys
from pathlib import Path

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]  # repository root
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / 'lightning'))
import lightning as L  # noqa: E402
import tiktoken  # noqa: E402

from bd3lm import BD3LM  # noqa: E402

CKPT = Path(sys.argv[1])  # a checkpoint of lightning/train.py
MASK = 50257
enc = tiktoken.get_encoding('gpt2')

ck = torch.load(CKPT, map_location='cpu', weights_only=False, mmap=True)
# mean step of the EMA average: decay (1+n)/(10+n) below 90k steps (weights ~ k^8: mean 0.9 n), 0.9999 after
N = ck['global_step']
T_EMA = 0.9 * N if N < 90000 else N - 10000
config = ck['hyper_parameters']['config']
sd = ck['state_dict']
torch.set_float32_matmul_precision('high')
L.seed_everything(config.seed)
init = BD3LM(config)
init_sd = init.state_dict()

train_names = [f'backbone.{n}' for n, p in init.backbone.named_parameters() if p.requires_grad]
train_names += [f'soft_mask.{n}' for n, p in init.soft_mask.named_parameters() if p.requires_grad]
ema = dict(zip(train_names, [e.float() for e in ck['ema']['shadow_params']]))

opt = ck['optimizer_states'][0]
opt_names = [f'backbone.{n}' for n, _ in init.backbone.named_parameters()] + \
            [f'soft_mask.{n}' for n, p in init.soft_mask.named_parameters() if p.requires_grad]
ids = [i for g in opt['param_groups'] for i in g['params']]
state = {n: opt['state'][i] for i, n in zip(ids, opt_names)}


def cos(a, b):
    return (a.flatten() @ b.flatten() / (a.norm() * b.norm() + 1e-30)).item()


def adam_dir(n):
    st = state[n]
    t = float(st['step'])
    m = st['exp_avg'].float() / (1 - 0.9 ** t)
    v = st['exp_avg_sq'].float() / (1 - 0.999 ** t)
    return -m / (v.sqrt() + 1e-8), (m.abs() / (v.sqrt() + 1e-12)).mean().item()


rows = []
agg = {k: 0.0 for k in ('dd', 'dp', 'pp', 'ud', 'uu', 'n')}
for n in train_names:
    w0, w, we = init_sd[n].float(), sd[n].float(), ema[n]
    d_rec, d_past = w - we, we - w0
    u, snr = adam_dir(n)
    proj = (d_rec.flatten() @ d_past.flatten() / (d_past.norm() + 1e-30)).item()
    rate = (proj / (N - T_EMA)) / (d_past.norm().item() / T_EMA + 1e-30)
    radial = (d_rec.flatten() @ w.flatten() / (w.norm() + 1e-30)).item() / (d_rec.norm().item() + 1e-30)
    rows.append((n, w.numel(), d_past.norm().item(), d_rec.norm().item(), cos(d_rec, d_past), rate, cos(u, d_rec), snr,
                 radial))
    agg['dd'] += (d_rec * d_rec).sum().item()
    agg['dp'] += (d_rec * d_past).sum().item()
    agg['pp'] += (d_past * d_past).sum().item()
    agg['ud'] += (u * d_rec).sum().item()
    agg['uu'] += (u * u).sum().item()
    agg['n'] += w.numel()

print(f'{"tensor":52s} {"numel":>9s} {"|Wema-W0|":>10s} {"|W-Wema|":>9s} {"cos_dir":>8s} {"rate":>6s} {"cos_adam":>9s} '
      f'{"snr":>5s} {"radial":>7s}')
for r in rows:
    n = r[0]
    if any(s in n for s in ('blocks.0.', 'blocks.6.', 'blocks.11.')) or 'blocks.' not in n:
        print(f'{n:52s} {r[1]:9d} {r[2]:10.3f} {r[3]:9.3f} {r[4]:8.3f} {r[5]:6.2f} {r[6]:9.3f} {r[7]:5.2f} {r[8]:7.3f}')

# by type, weighted by numel
print('\nby parameter type (numel-weighted means):')
kinds = {
    'embedding': lambda n: 'vocab_embed' in n,
    'output linear': lambda n: 'output_layer.linear' in n,
    'mlp (12 blocks)': lambda n: '.mlp.' in n,
    'adaLN (all)': lambda n: 'adaLN' in n,
    'attention qkvo': lambda n: 'attn.attn.' in n,
    'conv (12 blocks)': lambda n: 'local_conv' in n,
    'norms': lambda n: 'norm' in n,
}
for k, f in kinds.items():
    sel = [r for r in rows if f(r[0])]
    wsum = sum(r[1] for r in sel)
    avg = lambda i: sum(r[i] * r[1] for r in sel) / wsum  # noqa: E731
    print(f'  {k:18s} cos_dir {avg(4):6.3f}  rate {avg(5):5.2f}  cos_adam {avg(6):6.3f}  snr {avg(7):5.2f}  radial {avg(8):6.3f}')

D = agg['n']
print(f'\nall {D:,} trainable weights: cos_dir {agg["dp"] / (agg["dd"] * agg["pp"]) ** 0.5:.3f}, '
      f'cos_adam {agg["ud"] / (agg["uu"] * agg["dd"]) ** 0.5:.3f}  (random directions: ~{1 / D ** 0.5:.4f})')
print(f'  |W_ema - W0| = {agg["pp"] ** 0.5:.1f}, |W - W_ema| = {agg["dd"] ** 0.5:.1f}')

# the [MASK] row vs frequent tokens
E0 = init_sd['backbone.net.vocab_embed.embedding']
E, Ee = sd['backbone.net.vocab_embed.embedding'].float(), ema['backbone.net.vocab_embed.embedding']
st = state['backbone.net.vocab_embed.embedding']
t = float(st['step'])
m = st['exp_avg'].float() / (1 - 0.9 ** t)
v = st['exp_avg_sq'].float() / (1 - 0.999 ** t)
row_snr = (m.abs() / (v.sqrt() + 1e-12)).mean(1)
row_grad = v.sum(1).sqrt()  # RMS gradient norm of the row over the last ~1000 steps

tokens = np.load(sys.argv[2], mmap_mode='r')  # OpenWebText training token stream (~/.cache/gpr_datasets/*_tokens.npy)
counts = np.bincount(np.asarray(tokens[:300_000_000]), minlength=50258).astype(np.float64)
freq = counts / counts.sum()
order = np.argsort(-freq)


def row_stats(i):
    d_rec, d_past = E[i] - Ee[i], Ee[i] - E0[i]
    return dict(norm=Ee[i].norm().item(), moved=(E[i] - E0[i]).norm().item(), cos_dir=cos(d_rec, d_past),
                snr=row_snr[i].item(), grad=row_grad[i].item())


print('\n[MASK] row:', {k: round(x, 4) for k, x in row_stats(MASK).items()})
for label, idx in (('top-20 frequent', order[:20]), ('freq rank 1k-2k', order[1000:2000]),
                   ('freq rank 10k-20k', order[10000:20000])):
    s = [row_stats(int(i)) for i in idx]
    print(f'{label:18s}', {k: round(float(np.mean([x[k] for x in s])), 4) for k in s[0]})
print('top-10 frequent tokens:', [(enc.decode([int(i)]), round(freq[i] * 100, 2), round(Ee[int(i)].norm().item(), 3))
                                  for i in order[:10]])
norms = Ee.norm(dim=1).numpy()
seen = counts > 0
print(f'Spearman(norm, log freq) over seen tokens: '
      f'{np.corrcoef(np.argsort(np.argsort(norms[seen])), np.argsort(np.argsort(freq[seen])))[0, 1]:.3f}')
print('mean EMA norm by frequency decile (most frequent first):',
      [round(float(norms[order[k * 5000:(k + 1) * 5000]].mean()), 3) for k in range(10)])
