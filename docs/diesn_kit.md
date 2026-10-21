# diesn_kit

Files reused from `~/draft_refine` (branch `diffusion-refiner`, working tree of 2026-10-11). Copy the contents of
this folder into the root of the new project: the imports (`dataset.`, `data.`, `diffusion.`, `paralesn.`, `models.`)
assume these packages sit at the project root.

| Folder | Content |
|---|---|
| `dataset/` | the provided dataset classes: `InMemoryDataset` (tokenization with tiktoken, cache in `~/.cache/gpr_datasets`), `StreamingDataset` (HF streaming, tokenized on the fly, validation filtered by hash), `OpenWebTextDataset`, `TinyStoriesDataset`, `synthetic.py` |
| `data/` | `tokenizer.py` (GPT-2 + [MASK] = 50257), `token_datasets.py` (memmapped token streams cut into chunks of seq_len + 1), `datamodule.py` (`TokenDataModule`, Lightning; its config is `data/config.py: DataConfig`), `openwebtext.py` (OWT split: last 100k documents as validation, derived from the cache in seconds), `prepare.py` + `prepare_data.py` (HF datasets to uint16 `.bin`) |
| `paralesn/` | the reference ParalESN (unchanged) |
| `diffusion/` | `diesn.py` (DiESN backbone, unchanged), `transparency_head.py` (SMDM soft masking, unchanged), `config.py` (`DiESNConfig`: the fields DIESN reads, with the previous defaults) |
| `models/` | `fast_esn.py` (`use_fast_esn(backbone)`: the same ParalESN in one Triton kernel, ~4x faster), `attention.py` (FlashAttention 2 / SDPA, `bidirectional_attention`), `gpt.py` (only `apply_rotary_emb`, imported by diesn.py) |
| `scripts/_bootstrap.py` | environment for this machine: `NCCL_P2P_LEVEL=SYS` (multi-GPU all-reduce 0.6 → 14 GB/s), expandable segments |
| `tests/test_kit.py` | imports, DiESN with the fast ESN = reference |

Usage:

```python
from omegaconf import OmegaConf
from diffusion.config import DiESNConfig
from diffusion.diesn import DIESN
from models.fast_esn import use_fast_esn

backbone = DIESN(OmegaConf.structured(DiESNConfig(hidden_size=768, n_blocks=12, esn_n_units=768, esn_n_heads=12)),
                 vocab_size=50258)
use_fast_esn(backbone)  # same buffers and state_dict keys
```

```bash
python -m data.prepare_data --dataset openwebtext   # prints the train / val token files
```

Pitfalls:
- Never `torch.compile` the reference ParalESN (wrong reservoir states): use `use_fast_esn` (an opaque custom op),
  or wrap the reference with `models.fast_esn.eager_reference` before compiling the blocks.
- A100 has no FP8; FlashAttention 3 does not help there.

Dependencies (from the old `pyproject.toml`): `torch==2.9.*` (cu129), `triton`, `accelerated-scan>=0.3.1`,
`flash-linear-attention>=0.5.0`, `omegaconf>=2.3`, `tiktoken`, `datasets`, `numpy`, `tqdm`, `pyarrow`,
`lightning>=2.6.5` (datamodule only), `flash_attn` 2.8.3 wheel (optional: SDPA fallback),
`pytest` (tests).
