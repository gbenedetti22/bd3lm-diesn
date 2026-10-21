"""A few samples of 1024 tokens from a checkpoint of lightning/train.py (EMA weights, semi-autoregressive first-hitting
sampler, nucleus 0.9, as evaluate.py), written to a text file.
Usage: final_samples.py CKPT OUT.txt [N]"""
import sys
import time
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parents[1]  # repository root
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / 'lightning'))
import scripts._bootstrap  # noqa: E402,F401
import lightning as L  # noqa: E402

from bd3lm import BD3LM  # noqa: E402

ckpt, out = sys.argv[1], Path(sys.argv[2])
n = int(sys.argv[3]) if len(sys.argv) > 3 else 4
torch.set_float32_matmul_precision('high')
model = BD3LM.load_from_checkpoint(ckpt, map_location='cpu', weights_only=False)
model.config.loader.eval_batch_size = n
model.config.sampling.num_sample_batches = 1
model.to(torch.device('cuda', 0)).eval()
L.seed_everything(1)
t0 = time.time()
samples = model.restore_model_and_sample(num_steps=model.T)
print(f'{len(samples)} samples in {time.time() - t0:.0f} s')
out.write_text('\n\n========\n\n'.join(samples))
