"""Role of ParalESN and of the attention inside DiESN: a checkpoint of lightning/train.py (EMA weights) on validation
sequences of OpenWebText.

1. Contributions. Per block, the norm of what each branch writes into the residual stream: gate1 * (ESN @ merge),
   gate1 * GELU(conv), gate2 * MLP; for the attention layer, gate * attention. Each is relative to the norm of the
   residual stream entering the block. Measured on the x0 pass (clean sequence, the context) and on the final x_t
   pass (soft masks; masked positions).
2. Ablations. NELBO with the same t and masks (fixed seeds), in the following cases:
     - a branch removed in every block, or in one block at a time;
     - the context between blocks removed from one module, while the rest still sees it: from the ESN (forward
       reservoir restarted at every block), from the attention (own block only) or from the convolution (no
       positions before the block).
   NELBO also per range of masking rate t (blocks with t in [0, 1/3), [1/3, 2/3), [2/3, 1]).
Usage: paralesn_role.py CKPT TOKENS.npy OUT_PREFIX [--per-block]
"""
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F

ROOT = Path(__file__).resolve().parents[1]  # repository root
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / 'lightning'))
import scripts._bootstrap  # noqa: E402,F401

from bd3lm import BD3LM  # noqa: E402
from diesn_bd3 import X0, BlockDiffAttention, BlockDiffConv, BlockDiffESN  # noqa: E402
from models.dit import block_diff_mask  # noqa: E402
from models.gpt import apply_rotary_emb  # noqa: E402

CKPT, TOKENS, PREFIX = sys.argv[1], sys.argv[2], sys.argv[3]
PER_BLOCK = '--per-block' in sys.argv
N_GLOBAL, N_BLOCK, N_CONTRIB, BS = 512, 256, 64, 32
BINS = (0.0, 1 / 3, 2 / 3, 1.0001)


# -- the three mixers of diesn_bd3.py with ablations: module.ablate = None | 'zero' | 'local' -----------------------

@torch.compiler.disable
@torch.no_grad()
def esn_forward(self, x, attention_mask=None, **kwargs):
    """BlockDiffESN.forward; 'local': the forward reservoir restarts at every block in both halves (no context from
    earlier blocks), as the backward one."""
    mode = getattr(self, 'ablate', None)
    res, mixer = self.layers[0].reservoir, self.layers[0].mixer
    B, L, D = x.shape
    bs, H = self.block_size, res.n_units
    nb = L // bs
    out_dtype = torch.get_autocast_dtype('cuda') if torch.is_autocast_enabled('cuda') else torch.float32
    if mode == 'zero':
        return torch.zeros(B, L, 2 * H, dtype=out_dtype, device=x.device)
    with torch.autocast('cuda', enabled=False):
        w_in = torch.cat([res.in_kernel.real, res.in_kernel.imag], dim=1)
        p = (x.float().reshape(B * L, D) @ w_in).view(B, L, 2 * H)
    states = self._scan(p, B * nb).view(B, nb, bs, 4 * H)
    if mode != 'local':
        if self.half == X0:
            whole = self._scan(p, B).view(B, nb, bs, 4 * H)
            states[..., :H] = whole[..., :H]
            states[..., 2 * H:3 * H] = whole[..., 2 * H:3 * H]
            self.x0_ctx = torch.complex(whole[:, :-1, -1, :H], whole[:, :-1, -1, 2 * H:3 * H])
        else:
            lam_pow = torch.cumprod(res.recurrent_kernel.to(torch.complex64).expand(bs, H), dim=0)
            carry = self.x0_ctx[:, :, None, :] * lam_pow
            states[:, 1:, :, :H] += carry.real
            states[:, 1:, :, 2 * H:3 * H] += carry.imag
    w_mix = torch.cat([mixer.weight.real, -mixer.weight.imag], dim=1)
    return torch.tanh(F.linear(states.view(B, L, 4 * H).to(out_dtype), w_mix.to(out_dtype),
                               mixer.bias.real.to(out_dtype)))


@torch.compiler.disable
def conv_forward(self, x):
    """BlockDiffConv.forward (F.conv1d path); 'local': zeros instead of the 2 x0 positions before the block."""
    mode = getattr(self, 'ablate', None)
    if mode == 'zero':
        return torch.zeros_like(x)
    B, D, L = x.shape
    bs, r = self.block_size, self.halo
    nb = L // bs
    x = x.transpose(1, 2)
    if self.half == X0:
        idx = torch.arange(0, L, bs, device=x.device)[:, None] + torch.arange(r, device=x.device)
        self.x0_ctx = F.pad(x, (0, 0, r, 0))[:, idx]
    left = torch.zeros_like(self.x0_ctx) if mode == 'local' else self.x0_ctx
    windows = torch.cat([left, x.reshape(B, nb, bs, D), x.new_zeros(B, nb, r, D)], dim=2).view(B * nb, bs + 2 * r, D)
    out = F.conv1d(windows.transpose(1, 2), self.weight, self.bias, groups=D)
    return out.view(B, nb, D, bs).permute(0, 2, 1, 3).reshape(B, D, L)


@torch.compiler.disable
def attn_forward(self, hidden_states, attention_mask=None, **kwargs):
    """BlockDiffAttention.forward; 'local': every block reads only itself (M_BD), in both halves."""
    mode = getattr(self, 'ablate', None)
    if mode == 'zero':
        return torch.zeros_like(hidden_states), None, None
    B, T, C = hidden_states.shape
    q, k, v = (lin(hidden_states).view(B, T, self.n_head, self.head_dim) for lin in (self.c_q, self.c_k, self.c_v))
    freqs = torch.outer(torch.arange(T, device=q.device, dtype=torch.float32), self.inv_freq)
    cos, sin = (f[None, :, None, :].to(q.dtype) for f in (freqs.cos(), freqs.sin()))
    q = F.rms_norm(apply_rotary_emb(q, cos, sin), (self.head_dim,))
    k = F.rms_norm(apply_rotary_emb(k, cos, sin), (self.head_dim,))
    pos = torch.arange(T, device=q.device)
    if mode == 'local':
        mask = (pos[:, None] // self.block_size) == (pos[None, :] // self.block_size)
    elif self.half == X0:
        self.x0_ctx = (k, v)
        mask = block_diff_mask(None, None, pos[:, None] + T, pos[None, :] + T, block_size=self.block_size, n=T)
    else:
        k, v = torch.cat([k, self.x0_ctx[0]], dim=1), torch.cat([v, self.x0_ctx[1]], dim=1)
        kv_pos = torch.arange(2 * T, device=q.device)
        mask = block_diff_mask(None, None, pos[:, None], kv_pos[None, :], block_size=self.block_size, n=T)
    y = F.scaled_dot_product_attention(q.transpose(1, 2), k.transpose(1, 2), v.transpose(1, 2), attn_mask=mask)
    return self.c_proj(y.transpose(1, 2).reshape(B, T, C)), None, None


BlockDiffESN.forward, BlockDiffConv.forward, BlockDiffAttention.forward = esn_forward, conv_forward, attn_forward

# -- model and data -------------------------------------------------------------------------------------------------

torch.set_float32_matmul_precision('high')
dev = torch.device('cuda', 0)
model = BD3LM.load_from_checkpoint(CKPT, map_location='cpu', weights_only=False)
model.to(dev).eval()
model._ema_to_device()
model.ema.copy_to(model._get_parameters())  # EMA weights, as in validation
backbone, net = model.backbone, model.backbone.net
blocks = list(net.blocks)
tokens = np.load(TOKENS)
seqs = torch.from_numpy(tokens[:len(tokens) // 1024 * 1024].astype(np.int64)).view(-1, 1024)
print(Path(CKPT).name, '| validation sequences', len(seqs))


def set_ablation(esn=None, conv=None, attn=None, mlp=None, only=None):
    """Ablation modes for the ESN, the convolution and the MLP of every block (or of block `only`) and the attention."""
    for i, blk in enumerate(blocks):
        on = only is None or i == only
        blk.esn.ablate = esn if on else None
        blk.local_conv.ablate = conv if on else None
        if mlp == 'zero' and on:
            blk.mlp.forward = lambda x: torch.zeros_like(x)
        elif 'forward' in blk.mlp.__dict__:
            del blk.mlp.forward
    net.attn.attn.ablate = attn


@torch.no_grad()
def nelbo(n, seed=0):
    """NELBO per token (ignore_bos) with t and masks fixed by the seed, overall and per range of t."""
    tot, cnt = np.zeros(len(BINS) - 1), np.zeros(len(BINS) - 1)
    all_tot = all_cnt = 0.0
    for i in range(0, n, BS):
        torch.manual_seed(seed + i)
        x0 = seqs[i:i + BS].to(dev)
        t = model._sample_t(x0.shape, dev, 1e-3, 1.0)
        loss = model._forward_pass_diffusion(x0, t=t, sampling_eps_min=1e-3, sampling_eps_max=1.0)
        mask = torch.ones_like(loss)
        mask[:, 0] = 0
        loss = loss * mask
        all_tot += loss.sum().item()
        all_cnt += mask.sum().item()
        for k in range(len(BINS) - 1):
            sel = (t >= BINS[k]) & (t < BINS[k + 1]) & (mask > 0)
            tot[k] += loss[sel].sum().item()
            cnt[k] += sel.sum().item()
    return all_tot / all_cnt, tot / cnt


# -- 1. contributions -----------------------------------------------------------------------------------------------

rec = {'on': False, 'pass': None, 'xt_calls': 0, 'xt': None}
acc = {}  # (pass, layer, metric) -> list of per-token values
orig_x0, orig_xt = backbone.x0_pass, backbone.xt_pass


def x0_pass(x0, sigma):
    rec['pass'] = 'x0'
    return orig_x0(x0, sigma)


def xt_pass(xt, sigma):
    rec['xt_calls'] += 1
    rec['pass'] = 'xt' if rec['xt_calls'] % 2 == 0 else 'xt_prelim'  # soft-masked pass = second x_t pass of a batch
    rec['xt'] = xt[0][..., 0] if isinstance(xt, tuple) else xt
    return orig_xt(xt, sigma)


backbone.x0_pass, backbone.xt_pass = x0_pass, xt_pass


def positions():
    return None if rec['pass'] == 'x0' else (rec['xt'] == model.mask_index)


def add(layer, metric, values):
    sel = positions()
    v = values if sel is None else values[sel]
    acc.setdefault((rec['pass'], layer, metric), []).append(v.float().flatten().cpu())


def block_hooks(i, blk):
    cache = {}
    blk.register_forward_pre_hook(lambda m, a: cache.__setitem__('x', a[0]))
    blk.adaLN_modulation.register_forward_hook(lambda m, a, o: cache.__setitem__('mods', o))
    blk.esn.register_forward_hook(lambda m, a, o: cache.__setitem__('esn', o))
    blk.local_conv.register_forward_hook(lambda m, a, o: cache.__setitem__('conv', o))
    blk.mlp.register_forward_hook(lambda m, a, o: cache.__setitem__('mlp', o))

    def done(m, a, out):
        if not rec['on'] or rec['pass'] == 'xt_prelim':
            return
        x = cache['x'].float()
        _, _, g1, _, _, g2 = cache['mods'].float().chunk(6, dim=-1)
        u_esn = g1[:, None] * (cache['esn'].float() @ blk.merge_matr.float())
        u_conv = g1[:, None] * F.gelu(cache['conv'].float().transpose(1, 2))
        u_mlp = g2[:, None] * cache['mlp'].float()
        nx = x.norm(dim=-1)
        name = f'block{i}'
        add(name, 'resid_in', nx)
        add(name, 'esn/resid', u_esn.norm(dim=-1) / nx)
        add(name, 'conv/resid', u_conv.norm(dim=-1) / nx)
        add(name, 'esn+conv/resid', (u_esn + u_conv).norm(dim=-1) / nx)
        add(name, 'mlp/resid', u_mlp.norm(dim=-1) / nx)
        add(name, 'cos(esn,conv)', F.cosine_similarity(u_esn, u_conv, dim=-1))
    blk.register_forward_hook(done)


for i, blk in enumerate(blocks):
    block_hooks(i, blk)
attn_cache = {}
net.attn.register_forward_pre_hook(lambda m, a: attn_cache.__setitem__('x', a[0]))
net.attn.adaLN_modulation.register_forward_hook(lambda m, a, o: attn_cache.__setitem__('mods', o))
net.attn.attn.register_forward_hook(lambda m, a, o: attn_cache.__setitem__('y', o[0]))


def attn_done(m, a, out):
    if not rec['on'] or rec['pass'] == 'xt_prelim':
        return
    x = attn_cache['x'].float()
    _, _, g = attn_cache['mods'].float().chunk(3, dim=-1)
    u = g[:, None] * attn_cache['y'].float()
    nx = x.norm(dim=-1)
    add('attention', 'resid_in', nx)
    add('attention', 'attn/resid', u.norm(dim=-1) / nx)
    add('attention', 'resid_out', out.float().norm(dim=-1))


net.attn.register_forward_hook(attn_done)

set_ablation()
rec['on'] = True
nelbo(N_CONTRIB, seed=10_000)
rec['on'] = False
crow = [dict(pass_=p, layer=layer, metric=metric, mean=torch.cat(v).mean().item())
        for (p, layer, metric), v in acc.items()]
contrib = pd.DataFrame(crow).rename(columns={'pass_': 'pass'})
contrib.to_csv(f'{PREFIX}_contributions.csv', index=False)
for p in ('x0', 'xt'):
    piv = contrib[contrib['pass'] == p].pivot(index='layer', columns='metric', values='mean')
    order = [f'block{i}' for i in range(len(blocks))] + ['attention']
    print(f'\n[contributions, {p} pass{" (masked positions)" if p == "xt" else ""}]')
    print(piv.reindex(order).round(3).to_string())

# -- 2. ablations ---------------------------------------------------------------------------------------------------

runs = [('baseline', {}),
        ('no ESN (every block)', dict(esn='zero')),
        ('no conv (every block)', dict(conv='zero')),
        ('no attention', dict(attn='zero')),
        ('no MLP (every block)', dict(mlp='zero')),
        ('ESN without context between blocks', dict(esn='local')),
        ('conv without context between blocks', dict(conv='local')),
        ('attention without context between blocks', dict(attn='local')),
        ('only attention carries context', dict(esn='local', conv='local')),
        ('only ESN carries context', dict(conv='local', attn='local')),
        ('no context between blocks', dict(esn='local', conv='local', attn='local'))]
rows, base = [], None
for name, kw in runs:
    set_ablation(**kw)
    v, bins = nelbo(N_GLOBAL)
    base = v if base is None else base
    rows.append(dict(ablation=name, block='all', nll=v, delta=v - base,
                     **{f'nll_t[{BINS[k]:.2f},{min(BINS[k + 1], 1):.2f})': b for k, b in enumerate(bins)}))
    print(f'{name:45s} nll {v:.4f}  delta {v - base:+.4f}  per t range {np.round(bins, 3)}', flush=True)
if PER_BLOCK:
    set_ablation()
    base_b, _ = nelbo(N_BLOCK)
    for i in range(len(blocks)):
        for what in ('esn', 'conv', 'mlp'):
            set_ablation(**{what: 'zero'}, only=i)
            v, _ = nelbo(N_BLOCK)
            rows.append(dict(ablation=f'no {what}', block=i, nll=v, delta=v - base_b))
        print(f'block {i:2d}: ' + '  '.join(f'{r["ablation"]} {r["delta"]:+.4f}' for r in rows[-3:]), flush=True)
set_ablation()
pd.DataFrame(rows).to_csv(f'{PREFIX}_ablations.csv', index=False)
print('saved', f'{PREFIX}_contributions.csv', f'{PREFIX}_ablations.csv')
