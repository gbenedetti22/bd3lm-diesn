"""
Token sources and fixed-length chunk datasets.

Two sources are supported:

- datasets prepared by `data/prepare.py` from a HuggingFace text dataset (used for TinyStories):
  <data_dir>/<name>/{train,val}.bin (uint16) + meta.json, one <|endoftext|> before every document,
  cut into chunks by `ChunkDataset`;
- token streams already tokenized and cached by the provided `dataset/` classes (InMemoryDataset, used for
  OpenWebText): `TokenFileDataset` is that class on the cached .npy file, which it chunks itself.

Both GPT and refiner consume chunks of `seq_len + 1` tokens: the GPT uses (x[:-1] -> x[1:]), the refiner
refines x[1:] using the teacher-forced GPT distributions computed from x[:-1].

A training run can use a part of the training stream (`document_cut`): the draft GPT the first half, the refiner and
the detector the second half, so that they learn from the draft GPT's errors on text it has never seen.
"""

import json
import os
from dataclasses import dataclass

import numpy as np
import torch
from torch.utils.data import Dataset

from dataset.in_memory_dataset import InMemoryDataset

EOT = 50256


@dataclass
class TokenRegion:
    tokens: np.ndarray  # 1D (memory-mapped) token array
    start: int
    end: int

    @property
    def num_tokens(self):
        return self.end - self.start


class ChunkDataset(Dataset):
    """Fixed-length chunks from a token region (evenly spaced when `max_chunks` limits their number)."""

    def __init__(self, region: TokenRegion, chunk_len: int, max_chunks: int = 0):
        self.region = region
        self.chunk_len = chunk_len
        usable = region.num_tokens - chunk_len
        assert usable > 0, f"token region too small ({region.num_tokens}) for chunks of {chunk_len}"
        self.stride = chunk_len
        n = usable // self.stride + 1
        if max_chunks and n > max_chunks:
            self.stride = usable // max_chunks
            n = max_chunks
        self.n = n

    def __len__(self):
        return self.n

    def __getitem__(self, i):
        s = self.region.start + i * self.stride
        return torch.from_numpy(np.asarray(self.region.tokens[s:s + self.chunk_len], dtype=np.int64))


class TokenFileDataset(InMemoryDataset):
    """
    The provided InMemoryDataset (dataset/in_memory_dataset.py) on a token stream it already tokenized and
    cached (.npy). The file is memory-mapped instead of read into RAM (the OpenWebText train stream is
    18 GB and every DDP rank / job would hold a copy). Items: {input_ids, attention_mask, labels} over
    non-overlapping windows of seq_len tokens; `token_chunks` collates them into (B, seq_len + 1) chunks.
    """

    def __init__(self, path: str, seq_len: int, split: str):
        self.path = os.path.expanduser(path)
        # the OpenWebText cache of the gpr project has no document delimiter (special_token=False)
        super().__init__(dataset=self.path, seq_len=seq_len, split=split, special_token=False, cache=False,
                         verbose=False)

    def _load_or_build_tokens(self, dataset_arg):
        return np.load(self.path, mmap_mode="c")  # copy-on-write: a writable view without reading the file


def token_chunks(items) -> torch.Tensor:
    """Collate dataset/ items (labels = input_ids shifted by one) into (B, seq_len + 1) token chunks."""
    return torch.stack([torch.cat([it["input_ids"], it["labels"][-1:]]) for it in items])


def document_cut(tokens, fraction: float, block: int = 1 << 20) -> int:
    """Position of the document boundary at a fraction of a token stream: just after the first <|endoftext|> at or
    after fraction * len(tokens) (0 and 1 give the ends of the stream), so that no document is split."""
    n = len(tokens)
    start = int(round(fraction * n))
    if start <= 0 or start >= n:
        return min(max(start, 0), n)
    for s in range(start, n, block):
        ends = np.flatnonzero(np.asarray(tokens[s:s + block]) == EOT)
        if len(ends):
            return s + int(ends[0]) + 1
    return n


def window_range(lo: int, hi: int, seq_len: int) -> range:
    """Indices of the TokenFileDataset windows (tokens [i seq_len, (i + 1) seq_len], inputs + last label) that lie in
    the tokens [lo, hi)."""
    return range(-(-lo // seq_len), max((hi - 1) // seq_len, -(-lo // seq_len)))


# -----------------------------------------------------------------------------
# Prepared datasets

def _memmap_bin(path, dtype):
    return np.memmap(path, dtype=dtype, mode="r")


def load_prepared(data_dir: str, name: str, train_tokens: int = 0):
    """(train_region, val_region, meta) of a dataset prepared by data/prepare.py."""
    root = os.path.join(data_dir, name)
    meta_path = os.path.join(root, "meta.json")
    if not os.path.exists(meta_path):
        raise FileNotFoundError(
            f"Dataset '{name}' is not prepared in {root}. Run: python -m data.prepare_data --dataset {name}")
    with open(meta_path) as f:
        meta = json.load(f)
    dtype = np.dtype(meta.get("dtype", "uint16"))
    train = _memmap_bin(os.path.join(root, "train.bin"), dtype)
    val = _memmap_bin(os.path.join(root, "val.bin"), dtype)
    n_train = min(int(train_tokens), len(train)) if train_tokens else len(train)
    meta = dict(meta, train_tokens_used=n_train)
    return TokenRegion(train, 0, n_train), TokenRegion(val, 0, len(val)), meta
