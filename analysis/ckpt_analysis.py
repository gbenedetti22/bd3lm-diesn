"""CPU analysis of the final checkpoint of diesn_smdm_bs16: parameters vs the initialization (same seed as
train.py), embedding norms ([MASK] rank), output layer, EMA, Adam second moments, transparency head."""
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
MASK, EOT = 50257, 50256
enc = tiktoken.get_encoding('gpt2')


def tok(i):
    return '[MASK]' if i == MASK else repr(enc.decode([i]))


ck = torch.load(CKPT, map_location='cpu', weights_only=False, mmap=True)
config = ck['hyper_parameters']['config']
sd = ck['state_dict']
print('step', ck['global_step'], 'train h', ck['train_time_s'] / 3600)

# initial weights: same construction as train.py
torch.set_float32_matmul_precision('high')
L.seed_everything(config.seed)
init = BD3LM(config)
init_sd = init.state_dict()

# 1. frozen buffers unchanged and identical to a fresh construction (=> same RNG stream => init reproduced)
buf_names = [n for n, _ in init.named_buffers() if n in sd]
mismatch = [n for n in buf_names if not torch.equal(sd[n], init_sd[n])]
print(f'\n[buffers] {len(buf_names)} persistent buffers, mismatching the fresh init: {len(mismatch)}', mismatch[:5])

# 2. trainable parameters: NaN/Inf, norm, relative change from init, EMA
train_names = [f'backbone.{n}' for n, p in init.backbone.named_parameters() if p.requires_grad]
train_names += [f'noise.{n}' for n, p in init.noise.named_parameters() if p.requires_grad]
train_names += [f'soft_mask.{n}' for n, p in init.soft_mask.named_parameters() if p.requires_grad]
ema = ck['ema']['shadow_params']
assert len(ema) == len(train_names), (len(ema), len(train_names))
for n, e in zip(train_names, ema):
    assert e.shape == sd[n].shape, n
print('ema num_updates', ck['ema']['num_updates'], 'decay', ck['ema']['decay'])
ema = {n: e.float() for n, e in zip(train_names, ema)}

bad = [n for n in train_names if not torch.isfinite(sd[n]).all() or not torch.isfinite(ema[n]).all()]
print(f'[finite] tensors with NaN/Inf (raw or EMA): {bad}')

print(f'\n{"parameter":60s} {"|W0|":>9s} {"|W|":>9s} {"|W_ema|":>9s} {"|W-W0|/|W0|":>12s} {"|W-Wema|/|W|":>13s}')
for n in train_names:
    w0, w, we = init_sd[n].float(), sd[n].float(), ema[n]
    n0, n1, ne = w0.norm().item(), w.norm().item(), we.norm().item()
    rel = (w - w0).norm().item() / max(n0, 1e-12)
    rel_ema = (w - we).norm().item() / max(n1, 1e-12)
    if any(s in n for s in ('blocks.0.', 'blocks.5.', 'blocks.11.')) or 'blocks.' not in n:
        print(f'{n:60s} {n0:9.3f} {n1:9.3f} {ne:9.3f} {rel:12.3f} {rel_ema:13.4f}')

# 3. input embedding rows
for label, E, E0 in (('raw', sd['backbone.net.vocab_embed.embedding'].float(), init_sd['backbone.net.vocab_embed.embedding']),
                     ('ema', ema['backbone.net.vocab_embed.embedding'], init_sd['backbone.net.vocab_embed.embedding'])):
    norms = E.norm(dim=1)
    order = torch.argsort(norms, descending=True)
    rank = (order == MASK).nonzero().item() + 1
    untouched = (E == E0).all(dim=1)
    print(f'\n[embedding {label}] |e([MASK])| = {norms[MASK]:.3f}, rank {rank} / {len(norms)}; |e(EOT)| = {norms[EOT]:.3f}')
    q = torch.quantile(norms, torch.tensor([0.0, 0.01, 0.1, 0.5, 0.9, 0.99, 1.0]))
    print('  norm quantiles 0/1/10/50/90/99/100%:', [round(x, 3) for x in q.tolist()])
    print(f'  init row norm: mean {E0.norm(dim=1).mean():.3f}; rows identical to init (token never seen): {int(untouched.sum())}')
    print('  top 15 by norm:', [(tok(i), round(norms[i].item(), 2)) for i in order[:15].tolist()])
    if label == 'ema':
        Eu = E / norms[:, None]
        cos = Eu @ Eu[MASK]
        cos[MASK] = -2
        nn_idx = torch.argsort(cos, descending=True)[:10]
        print('  nearest to [MASK] (cosine):', [(tok(i), round(cos[i].item(), 3)) for i in nn_idx.tolist()])
        seen = ~untouched
        seen[MASK] = False
        print(f'  cosine([MASK], seen tokens): mean {cos[seen].mean():.3f}, max {cos[seen].max():.3f}')
        # how the average token embedding compares: soft mask mixes (1-lam) e_MASK + lam * sum_k p_k e_k
        mean_tok = E[seen].mean(0)
        print(f'  |mean of seen token embeddings| = {mean_tok.norm():.3f}; mean |e_tok| over seen = {norms[seen].mean():.3f}')

# 4. output layer
W = ema['backbone.net.output_layer.linear.weight']
b = ema['backbone.net.output_layer.linear.bias']
wn = W.norm(dim=1)
print(f'\n[output layer ema] |W[MASK]| = {wn[MASK]:.4f}, bias[MASK] = {b[MASK]:.4f}; row norm quantiles 50/99/100%:',
      [round(x, 3) for x in torch.quantile(wn, torch.tensor([0.5, 0.99, 1.0])).tolist()])
print('  highest bias:', [(tok(i), round(b[i].item(), 2)) for i in torch.argsort(b, descending=True)[:12].tolist()])

# 5. AdaLN: time_conditioning=False -> sigma = 0 -> c is a constant: the modulation is a constant per block
net = init.backbone.net
net.load_state_dict({k[len('backbone.net.'):]: v for k, v in ema.items() if k.startswith('backbone.net.')}, strict=False)
with torch.no_grad():
    c = torch.nn.functional.silu(net.sigma_map(torch.zeros(1)))
    print('\n[AdaLN, EMA] per block: mean |gate1| (ESN+conv), mean |gate2| (MLP), mean |scale1|, |norm1.w| mean')
    for i, blk in enumerate(net.blocks):
        sh1, sc1, g1, sh2, sc2, g2 = blk.adaLN_modulation(c).chunk(6, dim=-1)
        print(f'  block {i:2d}: gate1 {g1.abs().mean():.3f}  gate2 {g2.abs().mean():.3f}  scale1 {sc1.abs().mean():.3f}'
              f'  norm1.w {blk.norm1.weight.abs().mean():.3f}  conv |w| {blk.local_conv.weight.norm():.2f}')
    sh, sc, g = net.attn.adaLN_modulation(c).chunk(3, dim=-1)
    print(f'  attention: gate {g.abs().mean():.3f} scale {sc.abs().mean():.3f}')

# 6. Adam second moments: RMS gradient per tensor (clipped gradients: clipping happens before optimizer.step)
opt = ck['optimizer_states'][0]
groups = opt['param_groups']
print('\n[optimizer] groups:', [(len(g['params']), g['lr']) for g in groups])
opt_names = [f'backbone.{n}' for n, _ in init.backbone.named_parameters()] + \
            [f'noise.{n}' for n, _ in init.noise.named_parameters()] + \
            [f'soft_mask.{n}' for n, p in init.soft_mask.named_parameters() if p.requires_grad]
flat_ids = [i for g in groups for i in g['params']]
assert len(flat_ids) == len(opt_names), (len(flat_ids), len(opt_names))
tot = 0.0
rows = []
for i, n in zip(flat_ids, opt_names):
    st = opt['state'].get(i)
    if st is None:
        rows.append((n, None))
        continue
    v = st['exp_avg_sq'].float()
    step = float(st['step'])
    vhat = v / (1 - 0.999 ** step)
    tot += vhat.sum().item()
    rows.append((n, (vhat.sum().item() ** 0.5, vhat.mean().item() ** 0.5, step)))
print(f'  estimated RMS global gradient norm (after clipping at 1.0): {tot ** 0.5:.3f}')
big = sorted([r for r in rows if r[1]], key=lambda r: -r[1][0])[:10]
print('  largest per-tensor RMS gradient norms:', [(n, round(x[0], 4)) for n, x in big])
print('  soft-mask head:', [(n, x) for n, x in rows if n.startswith('soft_mask')])

# 7. transparency head
sm = init.soft_mask
sm.load_state_dict({k[len('soft_mask.'):]: v for k, v in sd.items() if k.startswith('soft_mask.')})
with torch.no_grad():
    print(f'\n[transparency raw] scale {sm.scale.item():.3f} centre {sm.centre.item():.3f} steepness {sm.steepness.item():.3f}')
    for H in (0.0, 0.5, 1, 2, 3, 4, 6, 8, 10):
        lam = sm.scale * torch.sigmoid(sm.steepness * (torch.tensor(-H) - sm.centre))
        print(f'  entropy {H:4.1f} nats -> lambda {lam.item():.3f}')
