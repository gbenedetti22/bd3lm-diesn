"""LightningDataModule serving fixed-length token chunks (seq_len + 1) for every training stage."""

import lightning as L
import numpy as np
from torch.utils.data import DataLoader, Subset

from data.token_datasets import (ChunkDataset, TokenFileDataset, TokenRegion, document_cut, load_prepared,
                                 token_chunks, window_range)


class TokenDataModule(L.LightningDataModule):
    def __init__(self, cfg):
        """cfg: data.config.DataConfig"""
        super().__init__()
        self.cfg = cfg
        self.train_ds = None
        self.val_ds = None
        self.val_region = None  # validation token stream (evaluation prompts / references)
        self.collate_fn = None
        self.meta = {}

    def setup(self, stage=None):
        if self.train_ds is not None:
            return
        c = self.cfg
        n_val = max(1, c.val_tokens // (c.seq_len + 1)) if c.val_tokens else 0
        assert len(c.train_slice) == 2 and 0.0 <= c.train_slice[0] < c.train_slice[1] <= 1.0, \
            f"data.train_slice must be [start, end] with 0 <= start < end <= 1, got {list(c.train_slice)}"
        if c.token_files:  # streams tokenized by the provided dataset/ classes (e.g. OpenWebText)
            train = TokenFileDataset(c.token_files["train"], c.seq_len, "train")
            val = TokenFileDataset(c.token_files["val"], c.seq_len, "validation")
            lo, hi = (document_cut(train.tokens_np, f) for f in c.train_slice)
            windows = window_range(lo, hi, c.seq_len)
            if c.train_tokens:
                windows = windows[:c.train_tokens // c.seq_len]
            self.train_ds = train if len(windows) == len(train) else Subset(train, windows)
            self.val_ds = val if not n_val or n_val >= len(val) else \
                Subset(val, np.linspace(0, len(val) - 1, n_val).astype(int).tolist())
            self.val_region = TokenRegion(val.tokens_np, 0, val.total_tokens)
            self.collate_fn = token_chunks
            self.meta = {"dataset": c.dataset, "source": dict(c.token_files), "train_tokens": train.total_tokens,
                         "val_tokens": val.total_tokens, "document_delimiter": train.special_token_id}
        else:  # prepared by data/prepare.py (e.g. TinyStories)
            train, val, self.meta = load_prepared(c.data_dir, c.dataset)
            lo, hi = (document_cut(train.tokens, f) for f in c.train_slice)
            hi = min(hi, lo + c.train_tokens) if c.train_tokens else hi
            self.train_ds = ChunkDataset(TokenRegion(train.tokens, lo, hi), c.seq_len + 1)
            self.val_ds = ChunkDataset(val, c.seq_len + 1, max_chunks=n_val)
            self.val_region = val
        self.meta.update(train_slice=list(c.train_slice), train_token_range=[lo, hi], train_chunks=len(self.train_ds),
                         val_chunks=len(self.val_ds), chunk_len=c.seq_len + 1)

    def train_dataloader(self):
        c = self.cfg
        return DataLoader(self.train_ds, batch_size=c.batch_size, shuffle=True, drop_last=True,
                          num_workers=c.num_workers, pin_memory=True, collate_fn=self.collate_fn,
                          persistent_workers=c.num_workers > 0)

    def val_dataloader(self):
        c = self.cfg
        return DataLoader(self.val_ds, batch_size=c.val_batch_size, shuffle=False, drop_last=False,
                          num_workers=min(c.num_workers, 4), pin_memory=True, collate_fn=self.collate_fn)
