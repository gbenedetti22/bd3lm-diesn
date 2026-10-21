"""Noise of one validation of the training run (val/nll: NELBO on [1e-3, 1], EMA weights, the 2048 validation
sequences of data.val_tokens): the same weights (step 25000) evaluated with different random t and masks."""
import sys
from pathlib import Path

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]  # repository root
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / 'lightning'))
import scripts._bootstrap  # noqa: E402,F401

from bd3lm import BD3LM, owt_datamodule  # noqa: E402

torch.set_float32_matmul_precision('high')
model = BD3LM.load_from_checkpoint(sys.argv[1], map_location='cpu',
                                   weights_only=False)
dev = torch.device('cuda', 0)
model.to(dev).eval()
model._ema_to_device()
model.ema.copy_to(model._get_parameters())  # EMA weights, as in validation
dm = owt_datamodule(model.config, 2097152)
dm.setup()
val = dm.collate_fn([dm.val_ds[i] for i in range(len(dm.val_ds))])
print('validation sequences', tuple(val.shape))

res = []
with torch.no_grad():
    for seed in range(6):
        torch.manual_seed(1000 + seed)
        tot, n = 0.0, 0.0
        for i in range(0, len(val), 32):
            x0 = val[i:i + 32].to(dev)
            losses = model._loss(x0, torch.ones_like(x0), sampling_eps_min=1e-3, sampling_eps_max=1.0)
            tot += losses.nlls.sum().item()
            n += losses.token_mask.sum().item()
        res.append(tot / n)
        print(f'seed {seed}: val/nll {res[-1]:.4f}')
res = np.array(res)
print(f'mean {res.mean():.4f}  std of one validation {res.std(ddof=1):.4f}  '
      f'std of the difference of two validations {res.std(ddof=1) * 2 ** 0.5:.4f}')
