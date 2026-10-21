"""CPU test of lightning/train.py: checkpoint_callbacks and resume_path with a tiny model (5 micro-batches per epoch,
accumulation 2 -> 3 optimizer steps per epoch), plus the state key of the step callback vs the real checkpoint."""
import os
import sys
import tempfile
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parents[1]  # repository root
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / 'lightning'))
import train  # noqa: E402  (lightning/train.py)
import lightning as L  # noqa: E402
from lightning.pytorch.demos.boring_classes import BoringModel  # noqa: E402
from torch.utils.data import DataLoader, Dataset  # noqa: E402


class Data(Dataset):
    def __len__(self):
        return 10

    def __getitem__(self, i):
        return torch.randn(32)


class Model(BoringModel):
    def train_dataloader(self):
        return DataLoader(Data(), batch_size=2)  # 5 micro-batches per epoch

    def val_dataloader(self):
        return DataLoader(Data(), batch_size=2)


class BarRecorder(L.Callback):
    """State of the training progress bar at every train batch end (after the progress bar callback)."""

    def __init__(self):
        self.states = []

    def on_train_batch_end(self, trainer, *_):
        bar = trainer.progress_bar_callback.train_progress_bar
        self.states.append((bar.desc, bar.n, bar.total, bar.initial))


def run(tmp, overrides, max_epochs, max_steps=-1, resume=None, recorder=None):
    config = train.load_config([f'trainer.{k}={v}' for k, v in overrides.items()] +
                               ([f'checkpointing.resume_ckpt_path={resume}'] if resume else []))
    ckpt_dir = Path(tmp) / 'checkpoints'
    bar = [train.TQDMProgressBar(refresh_rate=1), recorder] if recorder else []
    trainer = L.Trainer(accelerator='cpu', devices=1, max_epochs=max_epochs, max_steps=max_steps,
                        accumulate_grad_batches=2, val_check_interval=4, check_val_every_n_epoch=None,
                        num_sanity_val_steps=0, logger=False, enable_progress_bar=bool(recorder),
                        enable_model_summary=False, callbacks=bar + train.checkpoint_callbacks(config, ckpt_dir))
    trainer.fit(Model(), ckpt_path=train.resume_path(config, ckpt_dir))
    files = sorted(os.listdir(ckpt_dir))
    last = os.readlink(ckpt_dir / 'last.ckpt') if (ckpt_dir / 'last.ckpt').is_symlink() else None
    return trainer, files, last


with tempfile.TemporaryDirectory() as tmp:
    # 3 optimizer steps per epoch; step checkpoints every 4 steps (12 = end of epoch 3, saved after its validation)
    t, files, last = run(tmp, dict(ckpt_every=4, ckpt_keep=3, ckpt_epochs=-1), max_epochs=4)
    print('[keep all epochs] global_step', t.global_step, files, 'last ->', last)
    assert [f for f in files if f.startswith('epoch')] == ['epoch0-step3.ckpt', 'epoch1-step6.ckpt',
                                                           'epoch2-step9.ckpt', 'epoch3-step12.ckpt']
    assert sorted(f for f in files if f.startswith('step')) == ['step12.ckpt', 'step4.ckpt', 'step8.ckpt']
    assert last == 'step12.ckpt'
    ep = torch.load(Path(tmp) / 'checkpoints/epoch1-step6.ckpt', weights_only=False)
    print('  epoch1-step6: epoch', ep['epoch'], 'global_step', ep['global_step'],
          'epoch progress', ep['loops']['fit_loop']['epoch_progress']['current'])
    # resume from an end-of-epoch checkpoint at a step multiple of ckpt_every (explicit path), three more epochs
    t2, files2, last2 = run(tmp, dict(ckpt_every=4, ckpt_keep=3, ckpt_epochs=-1), max_epochs=7,
                            resume=Path(tmp) / 'checkpoints/epoch3-step12.ckpt')
    print('[resume from epoch3] global_step', t2.global_step, files2, 'last ->', last2)
    assert t2.global_step == 21 and not [f for f in files2 if '-v' in f], 'duplicate checkpoint after resuming'
    assert sorted(f for f in files2 if f.startswith('step')) == ['step12.ckpt', 'step16.ckpt', 'step20.ckpt']
    assert [f for f in files2 if f.startswith('epoch')][-3:] == ['epoch4-step15.ckpt', 'epoch5-step18.ckpt',
                                                                 'epoch6-step21.ckpt']
    assert last2 == 'step20.ckpt'
    # resume from last.ckpt (default), one more epoch: no duplicate of step20
    t3, files3, last3 = run(tmp, dict(ckpt_every=4, ckpt_keep=3, ckpt_epochs=-1), max_epochs=8)
    print('[resume from last.ckpt] global_step', t3.global_step, files3, 'last ->', last3)
    assert not [f for f in files3 if '-v' in f], 'versioned copy'
    assert sorted(f for f in files3 if f.startswith('step')) == ['step16.ckpt', 'step20.ckpt', 'step24.ckpt']
    assert 'epoch7-step24.ckpt' in files3 and last3 == 'step24.ckpt'
    try:
        train.resume_path(train.load_config(['checkpointing.resume_ckpt_path=/nonexistent.ckpt']), Path(tmp))
    except FileNotFoundError as e:
        print('[missing resume path] ', e)

with tempfile.TemporaryDirectory() as tmp:
    t, files, last = run(tmp, dict(ckpt_every=2, ckpt_keep=3, ckpt_epochs=2), max_epochs=4)
    print('[keep 2 epochs]', files, 'last ->', last)
    assert [f for f in files if f.startswith('epoch')] == ['epoch2-step9.ckpt', 'epoch3-step12.ckpt']

with tempfile.TemporaryDirectory() as tmp:
    t, files, last = run(tmp, dict(ckpt_every=2, ckpt_keep=3, ckpt_epochs=0), max_epochs=2)
    print('[no epoch checkpoints]', files, 'last ->', last)
    assert not [f for f in files if f.startswith('epoch')]

with tempfile.TemporaryDirectory() as tmp:
    # training stopped by max_steps in the middle of epoch 1 (steps 4-6): no epoch checkpoint for it
    t, files, last = run(tmp, dict(ckpt_every=4, ckpt_keep=3, ckpt_epochs=-1), max_epochs=50, max_steps=5)
    print('[max_steps mid-epoch]', files, 'last ->', last)
    assert [f for f in files if f.startswith('epoch')] == ['epoch0-step3.ckpt']
    # resume from step4.ckpt (micro-batch 2 of epoch 1) with the progress bar: epoch, total and position of the bar
    rec = BarRecorder()
    t, files, last = run(tmp, dict(ckpt_every=4, ckpt_keep=3, ckpt_epochs=-1), max_epochs=3,
                         resume=Path(tmp) / 'checkpoints/step4.ckpt', recorder=rec)
    print('[resume mid-epoch, TQDM bar] (desc, n, total, initial):', rec.states[:2], '...', rec.states[-1])
    assert [f for f in files if f.startswith('epoch')] == ['epoch0-step3.ckpt', 'epoch1-step6.ckpt',
                                                          'epoch2-step9.ckpt']

# the step callback of the real configuration restores the ModelCheckpoint state of the 25k checkpoint
REAL = ROOT / 'runs/diesn_smdm_bs16/snapshots/step25000.ckpt'  # a checkpoint saved before checkpoint_callbacks
if not REAL.exists():
    print('ALL OK (state key check skipped: no', REAL, ')')
    sys.exit(0)
ck = torch.load(REAL, map_location='cpu', weights_only=False, mmap=True)
cbs = [c for c in train.checkpoint_callbacks(train.load_config([]), ROOT / 'runs/diesn_smdm_bs16/checkpoints')
       if isinstance(c, train.ModelCheckpoint)]
print('\nstate keys in the 25k checkpoint:', [k for k in ck['callbacks'] if 'ModelCheckpoint' in k])
print('step callback state key:          ', cbs[0].state_key)
print('epoch callback state key:         ', cbs[1].state_key)
assert cbs[0].state_key in ck['callbacks']
print('saved state of the step callback:', {k: v for k, v in ck['callbacks'][cbs[0].state_key].items()
                                            if k in ('best_k_models', 'last_model_path', 'dirpath')})
print('ALL OK')
