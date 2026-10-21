# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## What this repository is

A thesis project built on top of the BD3LM codebase (Block Diffusion, arXiv:2503.09573, kuleshov-group/bd3lms). The
DiT Transformer backbone of BD3LM is replaced by **DiESN** (ParalESN reservoirs + softmax attention,
`diffusion/diesn.py`), optionally with **SMDM soft masking** (IBM's `TransparencyHead`, `diffusion/transparency_head.py`).
Experiments train on OpenWebText and compare BD3LM (DiT), BD3LM + DiESN, and BD3LM + DiESN + soft masking.

All new work lives in `lightning/`; see `lightning/README.md` (Italian) for the full design, run commands and metric
definitions.

## Ground rules from the user

- Do not modify pre-existing files (original BD3LM code, `diffusion/`, `models/`, `paralesn/`, `data/`, `dataset/`,
  `configs/`, `scripts/`) without explicit permission. `diffusion/diesn.py` must never be changed. New code goes in
  `lightning/`.
- Port BD3LM faithfully: same method names and logic as `diffusion.py`; mark deviations with `# lightning:` comments.
  Ask before design deviations from BD3LM or SMDM.
- No KV caching for DiESN (deliberately deferred).
- Never launch long trainings. Give `uv` commands to run inside tmux (see `lightning/README.md`). Short smoke runs
  (a few steps) are fine.
- Results are reported as exactly one table row per experiment with only: PPL | MAUVE | Tempo di addestramento |
  Tempo di inferenza. No other metrics.
- The user writes in Italian and wants explicit explanations in Italian.
- Profile/measure before claiming something is slow. Do not freeze live parameters to work around infrastructure
  issues.

## Commands

Environment is managed with `uv` (Python 3.11, torch 2.9 cu129). Hydra is **not** installed, so the original
`main.py` / `diffusion.py` / `configs/` are reference code only and do not run here.

```bash
uv sync                                                      # install
uv run pytest -q lightning/test_bd3lm.py                     # BD3LM/DiESN tests (need CUDA)
uv run pytest -q lightning/test_bd3lm.py -k test_sampling    # single test
uv run pytest -q tests/test_kit.py                           # kit tests (DiESN fast ESN = reference)
uv run python lightning/train.py name=<run> algo.backbone=diesn [key=value ...]
uv run python lightning/evaluate.py runs/<run>/checkpoints/last.ckpt "<row name>"   # appends to runs/results.md
uv run python -m data.prepare_data --dataset openwebtext    # rebuild tokenized OWT cache (~/.cache/gpr_datasets)
```

- `lightning/` has no `__init__.py` on purpose (it would shadow the `lightning` package). Run its scripts as files from
  the repo root (`python lightning/train.py`), never with `-m`. They insert the repo root into `sys.path` and import
  `scripts._bootstrap` (sets `NCCL_P2P_LEVEL=SYS` and CUDA expandable segments; required for fast multi-GPU here).
- Config is `lightning/config.yaml` (OmegaConf), overridden by `key=value` CLI args via `OmegaConf.from_dotlist`.
  Main switches: `algo.backbone=dit|diesn`, `algo.soft_masking=none|smdm` (smdm is DiESN only), `block_size` (16).
- Rerunning the same `train.py` command resumes from `runs/<name>/checkpoints/last.ckpt` (or
  `checkpointing.resume_ckpt_path`) and the same wandb run. Length: `trainer.max_steps`, or `trainer.max_epochs` with
  `trainer.max_steps=-1`. The progress bar is tqdm (rich's bar does not reach the wandb logs).
- One training at a time: each uses all visible GPUs (4× A100, DDP, global batch 512 × 1024 tokens, 64 per GPU with
  gradient accumulation). Evaluation needs one GPU.

## Architecture

### `lightning/bd3lm.py` — `BD3LM(L.LightningModule)`
Port of `Diffusion` in `diffusion.py` restricted to `algo=bd3lm`: SUBS parameterization, loglinear noise
(`noise_schedule.py`), antithetic sampling, clipped-schedule search (`var_min`), EMA, `metrics.Metrics`, and the
semi-autoregressive first-hitting sampler (`_semi_ar_sampler`, no KV cache). Training uses the BD3LM trick of one
forward on the concatenation `[x_t, x0]` with `block_diff_mask`. The loss computes `log p(x0|x_t)` only on masked
positions (`_subs_log_p_theta`); the full SUBS stays for sampling. `compile_backbone()` torch.compiles the per-token
part of the blocks for both backbones. Tokenizer is GPT-2 via tiktoken (`data/tokenizer.py`): EOT 50256,
`[MASK]` 50257, vocab 50258.

### `lightning/diesn_bd3.py` — `DiESNBD3`
Wraps the unmodified `DIESN` with the same interface as `models/dit.py: DIT`. Only DiESN's three sequence-mixing
modules are replaced (by `_BlockDiffMixer` subclasses sharing the same parameters, so the `state_dict` is identical
to `DIESN`'s); everything else in `DiESNBlock.forward` runs unchanged. Each mixer implements BD3LM's
`block_diff_mask` (bidirectional inside a block; noisy block b sees clean blocks < b; clean block b sees clean ≤ b):
- **ParalESN**: forward reservoir carries clean context across blocks; backward reservoir restarts each block; the
  noisy block starts from the clean state at the end of block b-1 via the linear correction `λ^(k+1)·h`.
- **Depthwise conv (k=5)**: left halo from `x0`, right halo zero; fused as shifted multiply-adds (`_depthwise_conv`).
- **Attention**: SDPA with `block_diff_mask`; requires `model.diesn.attn_impl: softmax` (GLA is rejected).
Training runs `x0_pass` once, then `xt_pass` reads its context; sampling is one pass over `[clean blocks, current block]`.

### Soft masking (`algo.soft_masking=smdm`)
In `_soft_masked_logits`: the clean `x0` half is computed once, then a no-grad preliminary `xt_pass` with binary
masks gives `p`, and `TransparencyHead` builds soft inputs `(1-λ)·e([MASK]) + λ·Σ top-k p·e(token)` for the final
`xt_pass`. Hyperparameters follow IBM's OWT recipe (`optim.sm_prob=0.8`, `optim.tran_head_lr=0.01`,
`algo.tran_head.init_scale=0.0`); the head's temperature is unused in top-k mode and frozen. With soft masking, DDP
uses `ddp_find_unused_parameters_true` (the head gets no gradient in non-soft steps). In sampling, the first step of
each block uses binary masks, later steps use the previous step's predictions.

### Invariants checked by `lightning/test_bd3lm.py`
Single block = original `DIESN`; training halves = sampling logits; block b never depends on `x0` of blocks ≥ b
(including through soft masks); compiled = eager; every parameter gets a gradient. Keep these passing after changes.

### Other pieces
- `models/fast_esn.py`: `use_fast_esn(backbone)` swaps the ParalESN scan for a Triton custom op (same state_dict).
  Never `torch.compile` the reference ParalESN directly (wrong reservoir states); use the fast op or
  `eager_reference`.
- `data/`: `TokenDataModule` over memmapped uint16 token files; OWT split = last 100k documents as validation.
- `lightning/train.py`: `TrainTimer` measures training-step wall-clock only and stores it in the checkpoint
  (`train_time_s`), which `evaluate.py` reports as training time.
- Checkpoints are ~2.6 GB and `/home` has little free space. Kept: the latest `trainer.ckpt_keep` (3) step checkpoints,
  `last.ckpt` (a link to the latest of them), and the end-of-epoch checkpoints `epoch<e>-step<s>.ckpt`
  (`trainer.ckpt_epochs`, -1 = all), which are the clean points to resume from (`train.py: checkpoint_callbacks`).
