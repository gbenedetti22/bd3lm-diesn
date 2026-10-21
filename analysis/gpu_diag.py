"""GPU diagnostics of the final checkpoint of diesn_smdm_bs16 (1 GPU, eager).

1. val/nll at step 25000 as the validations during training: same 2048 validation sequences (data.val_tokens),
   NELBO on [1e-3, 1], EMA weights (the training run stopped at max_steps before its validation at step 25000).
2. Paired evaluation, same validation sequences, same t and masks for every weight setting (fixed seeds):
   W(alpha) = W_ema + alpha (W - W_ema): alpha 0 = EMA, 1 = raw weights at step 25000, > 1 = further along the
   direction of the last ~2.5k steps. Also the EMA weights with binary masks (soft masking off) as a check.
3. Soft masks: lambda and entropy of the preliminary prediction at the masked positions.
4. Norm of the residual stream after every block (x_t pass): masked vs clean positions.
5. A few samples (EMA, first-hitting semi-AR sampler, nucleus 0.9).
"""
import sys
import time
from pathlib import Path

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]  # repository root
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / 'lightning'))
import scripts._bootstrap  # noqa: E402,F401
import lightning as L  # noqa: E402
from omegaconf import OmegaConf  # noqa: E402

from bd3lm import BD3LM, owt_datamodule  # noqa: E402

CKPT, OUT = Path(sys.argv[1]), Path(sys.argv[2])  # checkpoint of lightning/train.py, samples file
torch.set_float32_matmul_precision('high')

model = BD3LM.load_from_checkpoint(CKPT, map_location='cpu', weights_only=False)
model.config.merge_with(OmegaConf.from_dotlist(['algo.clip_search_widths=[]', 'loader.eval_batch_size=32',
                                                'model.attn_backend=sdpa']))
config = model.config
dev = torch.device('cuda', 0)
datamodule = owt_datamodule(config, 2097152)  # the validation subset of the training run (2048 chunks)

# 1. val/nll at step 25000 (Lightning validation loop of the module: EMA weights)
L.seed_everything(config.seed)
trainer = L.Trainer(accelerator='gpu', devices=1, precision='32-true', logger=False, enable_checkpointing=False,
                    inference_mode=False, enable_progress_bar=False)
t0 = time.time()
res = trainer.validate(model, datamodule=datamodule, verbose=False)[0]
print(f'[1] val at step 25000 (EMA, 2048 seq): nll {res["val/nll"]:.4f}  ppl {res["val/ppl"]:.2f}  ({time.time() - t0:.0f} s)')

# weight settings
model.to(dev).eval()
params = [p for p in model._get_parameters() if p.requires_grad]
raw = [p.detach().clone() for p in params]
ema = [s.to(dev) for s in model.ema.shadow_params]


def set_weights(alpha):
    with torch.no_grad():
        for p, r, e in zip(params, raw, ema):
            p.copy_(e + alpha * (r - e))


datamodule.setup()
val = datamodule.collate_fn([datamodule.val_ds[i] for i in range(0, 2048, 4)])  # 512 sequences
print('paired set:', tuple(val.shape))


@torch.no_grad()
def nelbo(seed=0, bs=32):
    tot, n = 0.0, 0.0
    for i in range(0, len(val), bs):
        torch.manual_seed(seed + i)  # same t and masks for every weight setting
        x0 = val[i:i + bs].to(dev)
        losses = model._loss(x0, torch.ones_like(x0), sampling_eps_min=1e-3, sampling_eps_max=1.0)
        tot += losses.nlls.sum().item()
        n += losses.token_mask.sum().item()
    return tot / n


print('\n[2] paired NELBO (512 seq, same t and masks):')
base = None
for alpha in (0.0, 0.5, 1.0, 1.5, 2.0, 3.0):
    set_weights(alpha)
    v = nelbo()
    base = v if base is None else base
    print(f'  alpha {alpha:3.1f}: nll {v:.4f}  ppl {np.exp(v):6.2f}  (vs EMA {v - base:+.4f})')
set_weights(0.0)
model.soft_masking = 'none'
v = nelbo()
print(f'  EMA, binary masks (soft masking off): nll {v:.4f}  ppl {np.exp(v):6.2f}  (vs soft {v - base:+.4f})')
model.soft_masking = 'smdm'

# 3. lambda and entropy at the masked positions; 4. residual stream norms
lam_log, ent_log = [], []
orig_head_forward = model.soft_mask.forward


def head_forward(input_ids, logits_prelim):
    out = orig_head_forward(input_ids, logits_prelim)
    masked = input_ids == model.mask_index
    lp = torch.log_softmax(logits_prelim[masked].float(), -1)
    ent_log.append(-(lp.exp() * lp).sum(-1).cpu())
    lam_log.append(out[1][..., 0][masked].float().cpu())  # weight of [MASK] = 1 - lambda
    return out


model.soft_mask.forward = head_forward
norms = {}
blocks = model.backbone.net.blocks
hooks = []
state = {'xt': None}


def make_hook(i):
    def hook(mod, inp, out):
        if model.backbone._mixers[0].half == 'xt' and state['xt'] is not None:
            masked = state['xt'] == model.mask_index
            nrm = out.float().norm(dim=-1)
            norms.setdefault(i, []).append((nrm[masked].mean().item(), nrm[~masked].mean().item()))
    return hook


for i, blk in enumerate(blocks):
    hooks.append(blk.register_forward_hook(make_hook(i)))
emb_out = {}


def emb_hook(mod, inp, out):
    if model.backbone._mixers[0].half == 'xt' and state['xt'] is not None:
        masked = state['xt'] == model.mask_index
        nrm = out.float().norm(dim=-1)
        emb_out.setdefault('e', []).append((nrm[masked].mean().item(), nrm[~masked].mean().item()))


hooks.append(model.backbone.net.vocab_embed.register_forward_hook(emb_hook))
orig_xt_pass = model.backbone.xt_pass


def xt_pass(xt, sigma):
    state['xt'] = xt[0][..., 0] if isinstance(xt, tuple) else xt
    return orig_xt_pass(xt, sigma)


model.backbone.xt_pass = xt_pass
set_weights(0.0)
with torch.no_grad():
    for i in range(0, 128, 32):
        torch.manual_seed(i)
        x0 = val[i:i + 32].to(dev)
        model._loss(x0, torch.ones_like(x0), sampling_eps_min=1e-3, sampling_eps_max=1.0)
for h in hooks:
    h.remove()
model.backbone.xt_pass = orig_xt_pass
model.soft_mask.forward = orig_head_forward
lam = 1 - torch.cat(lam_log)
ent = torch.cat(ent_log)
print('\n[3] soft masks at the masked positions (EMA):')
print('  lambda quantiles 5/25/50/75/95%:', [round(x, 3) for x in torch.quantile(lam, torch.tensor([.05, .25, .5, .75, .95])).tolist()])
print('  entropy (nats) quantiles 5/25/50/75/95%:', [round(x, 2) for x in torch.quantile(ent, torch.tensor([.05, .25, .5, .75, .95])).tolist()])
print('\n[4] residual stream norm, x_t pass (rows: prelim binary pass and soft pass mixed), masked | clean:')
e = np.array(emb_out['e'])
print(f'  embedding: {e[:, 0].mean():7.2f} | {e[:, 1].mean():7.2f}')
for i in sorted(norms):
    a = np.array(norms[i])
    print(f'  block {i:2d}: {a[:, 0].mean():7.2f} | {a[:, 1].mean():7.2f}')

# 5. samples
L.seed_everything(1)
config.sampling.num_sample_batches = 1
t0 = time.time()
model.config.loader.eval_batch_size = 4
samples = model.restore_model_and_sample(num_steps=model.T)
print(f'\n[5] {len(samples)} samples of 1024 tokens in {time.time() - t0:.0f} s')
OUT.write_text('\n\n=====\n\n'.join(samples))
print('samples written to', OUT)
