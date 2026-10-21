"""
Evaluate a checkpoint of lightning/train.py on the four metrics of the comparison and append its row to the results
table. From the project root:

    uv run python lightning/evaluate.py runs/diesn_bs16/checkpoints/last.ckpt "DiESN" [key=value ...]

  PPL        exp(NELBO per token) on the OpenWebText validation split (the last 100k documents; data.val_tokens
             tokens, 0 = all of it), EMA weights, first token excluded (ignore_bos): the bound that BD3LM reports
             (mode=ppl_eval of main.py).
  MAUVE      sampling.num_sample_batches x loader.eval_batch_size sequences generated as in mode=sample_eval of
             main.py (restore_model_and_sample: EMA weights, semi-autoregressive first-hitting sampler, nucleus
             sampling.nucleus_p) against as many human sequences of the validation split with the same number of
             tokens, mauve-text with GPT-2 large features.
  Training   hours of training steps (train.py: TrainTimer, stored in the checkpoint).
  Inference  seconds per generated sequence of model.length tokens: 1 GPU, batches of loader.eval_batch_size,
             no KV cache.
The key=value arguments override the configuration stored in the checkpoint (defaults below: EVAL_OVERRIDES).
"""

import argparse
import json
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))  # project root: diffusion/, models/, data/, metrics.py
import scripts._bootstrap  # noqa: E402,F401  (CUDA expandable segments)

import lightning as L  # noqa: E402
import mauve  # noqa: E402
import torch  # noqa: E402
from omegaconf import OmegaConf  # noqa: E402

from bd3lm import BD3LM, owt_datamodule  # noqa: E402
from data.token_datasets import ChunkDataset  # noqa: E402

EVAL_OVERRIDES = [
    "data.val_tokens=0",  # PPL on the whole validation split
    "algo.clip_search_widths=[]",  # only the NELBO in validation_step (the clipped intervals are a training aid)
    "model.attn_backend=sdpa",  # dit: no FlexAttention compilation for evaluation
    "loader.eval_batch_size=50",  # PPL and sampling batch
    "sampling.num_sample_batches=20",  # 1000 generated sequences
]
HEADER = "| Esperimento | PPL | MAUVE | Tempo di addestramento | Tempo di inferenza |\n|---|---|---|---|---|\n"


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("ckpt")
    p.add_argument("name", help="row label in the results table")
    p.add_argument("overrides", nargs="*", help="key=value")
    p.add_argument("--results", default="runs/results.md")
    p.add_argument("--mauve_model", default="gpt2-large", help="MAUVE featurizer")
    a = p.parse_args(argv)
    torch.set_float32_matmul_precision("high")

    model = BD3LM.load_from_checkpoint(a.ckpt, map_location="cpu", weights_only=False)  # as main.py
    model.config.merge_with(OmegaConf.from_dotlist(EVAL_OVERRIDES + a.overrides))
    config = model.config
    L.seed_everything(config.seed)
    datamodule = owt_datamodule(config, config.data.val_tokens)

    # PPL: validation_step of the module (the NELBO), EMA weights
    # inference_mode=False, as the validation inside Trainer.fit: the metrics reset in it are updated by the sampler
    trainer = L.Trainer(accelerator="gpu", devices=1, precision="32-true", logger=False, enable_checkpointing=False,
                        inference_mode=False)
    ppl = trainer.validate(model, datamodule=datamodule)[0]["val/ppl"]

    # samples and inference time
    device = torch.device("cuda", torch.cuda.current_device())
    model.to(device).eval()
    n_batches = config.sampling.num_sample_batches
    config.sampling.num_sample_batches = 1  # warm-up (kernels, allocator), not timed
    model._sample(seqlen=2 * model.block_size, num_steps=model.T, batch_size_per_gpu=2)
    config.sampling.num_sample_batches = n_batches
    torch.cuda.synchronize(device)
    t0 = time.perf_counter()
    generated = model.restore_model_and_sample(num_steps=model.T)
    torch.cuda.synchronize(device)
    inference_s = (time.perf_counter() - t0) / len(generated)

    # MAUVE: the generated tokens after the BOS against human chunks of model.length - 1 tokens
    length = config.model.length
    chunks = ChunkDataset(datamodule.val_region, length, max_chunks=len(generated))
    human = model.tokenizer.decode_batch([chunks[i][1:] for i in range(len(chunks))])
    mauve_score = mauve.compute_mauve(p_text=human, q_text=generated, featurize_model_name=a.mauve_model,
                                      device_id=device.index, max_text_length=length, verbose=False,
                                      seed=config.seed).mauve

    train_h = model.train_time_s / 3600  # restored from the checkpoint by BD3LM.on_load_checkpoint
    row = f"| {a.name} | {ppl:.2f} | {mauve_score:.3f} | {train_h:.1f} h | {inference_s:.2f} s |\n"
    results = Path(a.results)
    results.parent.mkdir(parents=True, exist_ok=True)
    new_table = not results.exists() or results.stat().st_size == 0
    with open(results, "a") as f:
        f.write((HEADER if new_table else "") + row)
    out = Path(a.ckpt).resolve().parent.parent / "eval_samples.json"
    out.write_text(json.dumps({"name": a.name, "ckpt": a.ckpt, "samples": generated}, indent=1))
    print(HEADER + row, end="")
    print(f"[evaluate] row appended to {results}; generated texts in {out}")


if __name__ == "__main__":
    main()
