"""Configuration read by data/datamodule.py: TokenDataModule (any object with these attributes works)."""

from dataclasses import dataclass, field
from typing import Dict


@dataclass
class DataConfig:
    dataset: str = "tinystories"  # dataset name (a dataset prepared in data_dir, unless token_files is set)
    data_dir: str = "data_cache"
    # token streams already tokenized by the dataset/ classes, {train: .npy, val: .npy}
    token_files: Dict[str, str] = field(default_factory=dict)
    seq_len: int = 512  # training sequence length (chunks of seq_len + 1 tokens)
    batch_size: int = 64  # per device
    val_batch_size: int = 64
    num_workers: int = 8
    val_tokens: int = 2_097_152  # validation budget (chunks evenly spaced over the validation split); 0 = all
    train_slice: list = field(default_factory=lambda: [0.0, 1.0])  # part of the training stream (fractions)
    train_tokens: int = 0  # 0 = the whole training slice; otherwise its first train_tokens tokens
