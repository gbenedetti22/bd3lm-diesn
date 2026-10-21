"""Norm of the [MASK] input embedding (EMA weights) at every end-of-epoch checkpoint: rank among the 50258 rows,
median row norm, cosine with the closest token. Reads only the embedding tensor of each checkpoint (mmap).
Usage: mask_norm_epochs.py OUT.csv CKPT [CKPT ...]"""
import re
import sys

import pandas as pd
import torch

MASK = 50257
rows = []
for path in sys.argv[2:]:
    ck = torch.load(path, map_location='cpu', weights_only=False, mmap=True)
    E = ck['ema']['shadow_params'][0].float()  # backbone.net.vocab_embed.embedding (first trainable parameter)
    assert E.shape == (50258, 768), E.shape
    norms = E.norm(dim=1)
    rank = int((norms > norms[MASK]).sum()) + 1
    Eu = E / norms[:, None]
    cos = Eu @ Eu[MASK]
    cos[MASK] = -2
    rows.append(dict(ckpt=path.split('/')[-1], step=ck['global_step'], mask_norm=norms[MASK].item(), mask_rank=rank,
                     median_norm=norms.median().item(), max_norm=norms.max().item(), max_cos_with_token=cos.max().item()))
    print(rows[-1], flush=True)
    del ck
df = pd.DataFrame(rows).sort_values('step')
df.to_csv(sys.argv[1], index=False)
print(df.round(3).to_string(index=False))
