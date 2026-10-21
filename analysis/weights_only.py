"""Weights-only copy of a checkpoint of lightning/train.py: model state_dict, EMA weights, configuration, clipped
schedule and training time (what BD3LM.load_from_checkpoint and lightning/evaluate.py read), without the optimizer and
loop states. Usage: weights_only.py CKPT OUT"""
import sys

import torch

ck = torch.load(sys.argv[1], map_location='cpu', weights_only=False, mmap=True)
drop = ('optimizer_states', 'lr_schedulers', 'loops', 'callbacks')
out = {k: v for k, v in ck.items() if k not in drop}
print('kept keys:', list(out), '| step', out['global_step'], '| train h', out['train_time_s'] / 3600)
torch.save(out, sys.argv[2])
