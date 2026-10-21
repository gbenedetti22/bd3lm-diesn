from typing import Any

from dataset.in_memory_dataset import InMemoryDataset


class OpenWebTextDataset(InMemoryDataset):
    def __init__(
        self,
        tokenizer: Any = "gpt2",
        seq_len: int = 1024,
        split: str = "train",
        dataset: str = "Skylion007/openwebtext",
        **kwargs,
    ):
        super().__init__(
            dataset=dataset,
            tokenizer=tokenizer,
            seq_len=seq_len,
            split=split,
            **kwargs,
        )