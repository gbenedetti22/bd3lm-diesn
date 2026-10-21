"""Depth profiles of DiESN from the EMA weights of a series of checkpoints of lightning/train.py.

  - AdaLN of every block, of the attention layer and of the output layer. With time_conditioning=False sigma is 0,
    so the modulation is a learned constant: shift, scale, gate per channel. Effective gain of the normalized input:
    |LayerNorm weight * (1 + scale)|.
  - Weight norms (Frobenius and spectral) of the MLPs, of the attention projections and of the convolutions.
  - Memory of the frozen reservoirs (identical in every checkpoint): |lambda|, half-life in tokens, fraction of the
    state left after one block of 16 tokens.
Usage: depth_profiles.py OUT.csv CKPT [CKPT ...]
"""
import math
import sys
from pathlib import Path

import pandas as pd
import torch
import torch.nn.functional as F

ROOT = Path(__file__).resolve().parents[1]  # repository root
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / 'lightning'))
import lightning as L  # noqa: E402

from bd3lm import BD3LM  # noqa: E402

ckpts = sys.argv[2:]
first = torch.load(ckpts[0], map_location='cpu', weights_only=False, mmap=True)
config = first['hyper_parameters']['config']
L.seed_everything(config.seed)
model = BD3LM(config)
names = [f'backbone.{n}' for n, p in model.backbone.named_parameters() if p.requires_grad]
names += [f'soft_mask.{n}' for n, p in model.soft_mask.named_parameters() if p.requires_grad]
net = model.backbone.net
bs = config.block_size

print('reservoir memory (frozen, the same in every checkpoint):')
for i, blk in enumerate(net.blocks):
    lam = blk.esn.layers[0].reservoir.recurrent_kernel.abs().double()
    half = math.log(0.5) / torch.log(lam)
    print(f'  block {i:2d}: |lambda| {lam.min():.3f}-{lam.max():.3f} (median {lam.median():.3f}), half-life '
          f'{half.min():.2f}-{half.max():.2f} tokens (median {half.median():.2f}), left after {bs} tokens: '
          f'max {lam.max() ** bs:.3f}, median {lam.median() ** bs:.4f}')


def rms(t):
    return t.float().pow(2).mean().sqrt().item()


rows = []
for path in ckpts:
    ck = torch.load(path, map_location='cpu', weights_only=False, mmap=True)
    ema = dict(zip(names, ck['ema']['shadow_params']))
    net.load_state_dict({k[len('backbone.net.'):]: v for k, v in ema.items() if k.startswith('backbone.net.')},
                        strict=False)
    step = ck['global_step']
    with torch.no_grad():
        c = F.silu(net.sigma_map(torch.zeros(1)))
        for i, blk in enumerate(net.blocks):
            sh1, sc1, g1, sh2, sc2, g2 = blk.adaLN_modulation(c).chunk(6, dim=-1)
            w1, w2 = blk.mlp[0].weight, blk.mlp[2].weight
            rows.append(dict(
                ckpt=Path(path).name, step=step, layer=f'block{i}', depth=i,
                shift1=rms(sh1), scale1=rms(sc1), gate1=rms(g1), shift2=rms(sh2), scale2=rms(sc2), gate2=rms(g2),
                gain1=(blk.norm1.weight * (1 + sc1)).abs().mean().item(),
                gain2=(blk.norm2.weight * (1 + sc2)).abs().mean().item(),
                mlp_w1=w1.norm().item(), mlp_w2=w2.norm().item(),
                mlp_w1_spec=torch.linalg.matrix_norm(w1.float(), 2).item(),
                mlp_w2_spec=torch.linalg.matrix_norm(w2.float(), 2).item(),
                conv_w=blk.local_conv.weight.norm().item()))
        sh, sc, g = net.attn.adaLN_modulation(c).chunk(3, dim=-1)
        a = net.attn.attn
        rows.append(dict(ckpt=Path(path).name, step=step, layer='attention', depth=len(net.blocks),
                         shift1=rms(sh), scale1=rms(sc), gate1=rms(g),
                         gain1=(net.attn.norm.weight * (1 + sc)).abs().mean().item(),
                         attn_q=a.c_q.weight.norm().item(), attn_k=a.c_k.weight.norm().item(),
                         attn_v=a.c_v.weight.norm().item(), attn_o=a.c_proj.weight.norm().item(),
                         attn_vo_spec=torch.linalg.matrix_norm((a.c_proj.weight @ a.c_v.weight).float(), 2).item()))
        sh, sc = net.output_layer.adaLN_modulation(c).chunk(2, dim=-1)
        rows.append(dict(ckpt=Path(path).name, step=step, layer='output', depth=len(net.blocks) + 1,
                         shift1=rms(sh), scale1=rms(sc),
                         gain1=(net.output_layer.norm_final.weight * (1 + sc)).abs().mean().item(),
                         out_w=net.output_layer.linear.weight.norm().item()))
    print('done', Path(path).name, flush=True)
    del ck

df = pd.DataFrame(rows)
df.to_csv(sys.argv[1], index=False)
last = df[df['step'] == df['step'].max()]
cols = ['layer', 'gate1', 'gate2', 'scale1', 'scale2', 'shift1', 'shift2', 'gain1', 'gain2', 'mlp_w1', 'mlp_w2',
        'mlp_w1_spec', 'mlp_w2_spec', 'conv_w']
print(last[cols].round(3).to_string(index=False))
