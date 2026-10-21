"""
Train BD3LM on OpenWebText with PyTorch Lightning (Trainer, ModelCheckpoint, WandbLogger). From the project root:

    uv run python lightning/train.py name=diesn_bs16 algo.backbone=diesn

Configuration: lightning/config.yaml, overridden by key=value arguments (as main.py with Hydra). Rerunning the same
command resumes from <out_dir>/<name>/checkpoints/last.ckpt (or checkpointing.resume_ckpt_path, e.g. an end-of-epoch
checkpoint) and continues the same wandb run. Length: trainer.max_steps, or trainer.max_epochs with
trainer.max_steps=-1.
"""

import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))  # project root: diffusion/, models/, data/, metrics.py
import scripts._bootstrap  # noqa: E402,F401  (NCCL P2P across the CPU sockets, CUDA expandable segments)

import lightning as L  # noqa: E402
import torch  # noqa: E402
from lightning.pytorch.callbacks import LearningRateMonitor, ModelCheckpoint, TQDMProgressBar  # noqa: E402
from lightning.pytorch.loggers import WandbLogger  # noqa: E402
from omegaconf import OmegaConf  # noqa: E402

from bd3lm import BD3LM, owt_datamodule  # noqa: E402


class TrainTimer(L.Callback):
    """Training time: wall-clock of the training steps (forward, backward, optimizer step; CUDA synchronized),
    without validation, checkpointing (ModelCheckpoint runs after the other callbacks) and waits for the data. Kept
    in the checkpoints (BD3LM.train_time_s), so it adds up across resumed runs."""

    def on_train_batch_start(self, trainer, pl_module, batch, batch_idx):
        self._t0 = time.perf_counter()

    def on_train_batch_end(self, trainer, pl_module, outputs, batch, batch_idx):
        torch.cuda.synchronize(pl_module.device)
        pl_module.train_time_s += time.perf_counter() - self._t0
        pl_module.log("trainer/time_h", pl_module.train_time_s / 3600, on_step=True, on_epoch=False)


class EpochCheckpoint(ModelCheckpoint):
    """ModelCheckpoint at the end of complete epochs only: Lightning also ends the epoch when training stops in the
    middle of it (max_steps), which would save a checkpoint named after an epoch that did not end."""

    def on_train_epoch_end(self, trainer, pl_module):
        if trainer.fit_loop.epoch_loop.batch_progress.is_last_batch:
            super().on_train_epoch_end(trainer, pl_module)


class SkipResumedStep(L.Callback):
    """No checkpoint at the step a run resumes from: the ModelCheckpoint state does not keep its last saved step, so
    resuming from step{s}.ckpt it would save step{s}-v1.ckpt at once (a copy taking one of the trainer.ckpt_keep)."""

    def on_train_start(self, trainer, pl_module):
        for callback in trainer.checkpoint_callbacks:
            callback._last_global_step_saved = trainer.global_step


def load_config(overrides):
    """lightning/config.yaml with the key=value overrides."""
    return OmegaConf.merge(OmegaConf.load(Path(__file__).with_name("config.yaml")), OmegaConf.from_dotlist(overrides))


def checkpoint_callbacks(config, ckpt_dir):
    """Checkpoint callbacks (configs/callbacks/checkpoint_every_n_steps.yaml keeps every checkpoint; here fewer:
    /home has little free space):
      - every trainer.ckpt_every optimizer steps, step{step}.ckpt: the latest trainer.ckpt_keep (top k on "step"),
        and last.ckpt, a link to the latest, for resuming;
      - at the end of every epoch, epoch{epoch}-step{step}.ckpt (epoch: Lightning's 0-based index, as the "epoch"
        logged to wandb): a point to resume from with the data loader at the start of an epoch (Lightning's samplers
        restart the epoch when resuming in the middle of it). The latest trainer.ckpt_epochs (-1: all, 0: none)."""
    t = config.trainer
    # the step callback keeps its arguments (monitor, mode, every_n_train_steps): same state key as in the
    # checkpoints already saved, whose ModelCheckpoint state (saved checkpoints, last.ckpt) it restores.
    # enable_version_counter=False: a step or an epoch end done again after resuming from an earlier checkpoint
    # overwrites its file (Lightning saves atomically) instead of writing an untracked -v1 copy next to it
    callbacks = [SkipResumedStep(),
                 ModelCheckpoint(dirpath=ckpt_dir, filename="step{step}", auto_insert_metric_name=False,
                                 every_n_train_steps=t.ckpt_every, monitor="step", mode="max", save_top_k=t.ckpt_keep,
                                 save_last="link", enable_version_counter=False)]
    if t.ckpt_epochs:
        keep_all = t.ckpt_epochs < 0
        # save_on_train_epoch_end: with check_val_every_n_epoch=None Lightning would save it after a validation;
        # no save_last: last.ckpt belongs to the step callback (a second one would write last-v1.ckpt)
        callbacks.append(EpochCheckpoint(dirpath=ckpt_dir, filename="epoch{epoch}-step{step}",
                                         auto_insert_metric_name=False, every_n_epochs=1, save_on_train_epoch_end=True,
                                         monitor=None if keep_all else "epoch", mode="max",
                                         save_top_k=-1 if keep_all else t.ckpt_epochs, enable_version_counter=False))
    return callbacks


def resume_path(config, ckpt_dir):
    """checkpointing.resume_ckpt_path if given (it must exist), else last.ckpt of the run if any (main.py)."""
    path = config.checkpointing.resume_ckpt_path
    if path is None:
        last = ckpt_dir / "last.ckpt"
        return str(last) if last.exists() else None
    if not Path(path).exists():
        raise FileNotFoundError(f"checkpointing.resume_ckpt_path: {path} does not exist")
    return str(path)


def main(argv=None):
    config = load_config(sys.argv[1:] if argv is None else argv)
    devices = torch.cuda.device_count() if config.trainer.devices == -1 else config.trainer.devices
    accumulate, rest = divmod(config.loader.global_batch_size, config.loader.batch_size * devices)
    assert accumulate >= 1 and rest == 0, "loader.global_batch_size must be a multiple of batch_size x GPUs"
    torch.set_float32_matmul_precision("high")
    L.seed_everything(config.seed)  # the same frozen reservoirs on every process

    model = BD3LM(config)
    if config.trainer.compile:
        model.compile_backbone()
    datamodule = owt_datamodule(config, config.data.val_tokens)

    run_dir = Path(config.out_dir) / config.name
    run_dir.mkdir(parents=True, exist_ok=True)
    ckpt_dir = run_dir / "checkpoints"
    logger = WandbLogger(project=config.wandb.project, name=config.name, id=f"{config.name}_{config.seed}",
                         resume="allow", save_dir=str(run_dir), mode=config.wandb.mode,
                         config=OmegaConf.to_container(config))
    strategy = "auto"
    if devices > 1:  # soft masking runs in a fraction optim.sm_prob of the steps: in the others the TransparencyHead
        # gets no gradient. As SMDM's scripts (strategy.find_unused_parameters=True); measured cost ~0.5% per step
        strategy = "ddp_find_unused_parameters_true" if config.algo.soft_masking != "none" else "ddp"
    trainer = L.Trainer(
        accelerator="gpu", devices=devices, strategy=strategy,
        precision="32-true",  # fp32 weights; the backbones run their blocks under bf16 autocast (as diffusion.py)
        max_steps=config.trainer.max_steps, max_epochs=config.trainer.max_epochs,
        accumulate_grad_batches=accumulate, gradient_clip_val=config.trainer.gradient_clip_val,
        val_check_interval=config.trainer.val_check_interval * accumulate,  # Lightning counts micro-batches
        check_val_every_n_epoch=None, log_every_n_steps=config.trainer.log_every_n_steps,
        num_sanity_val_steps=config.trainer.num_sanity_val_steps,
        # TQDM progress bar: wandb shows it in the run's logs (with rich installed Lightning's default is the rich
        # progress bar, which does not reach them)
        callbacks=[TrainTimer(), LearningRateMonitor(logging_interval="step"),
                   TQDMProgressBar(refresh_rate=config.trainer.progress_bar_refresh_rate),
                   *checkpoint_callbacks(config, ckpt_dir)],
        logger=logger, default_root_dir=run_dir)
    # weights_only=False: the checkpoint holds the OmegaConf configuration (as main.py does with its checkpoints)
    trainer.fit(model, datamodule=datamodule, ckpt_path=resume_path(config, ckpt_dir), weights_only=False)


if __name__ == "__main__":
    main()
