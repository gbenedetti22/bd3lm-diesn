"""
Streaming counterpart of InMemoryDataset for corpora too large to download or keep in RAM
(e.g. FineWeb-Edu). Documents are read lazily from a HuggingFace streaming dataset (or any
iterable of strings), tokenized on the fly in multi-threaded batches, joined with the document
delimiter token and cut into (x, y) sequences exactly like InMemoryDataset.__getitem__.

Train/validation split for datasets without a validation split: the first `val_num_docs`
documents of the (unshuffled) stream form the validation set, which is small and is returned as
an InMemoryDataset; the training stream drops every document whose text hash is in that set.
Filtering by hash is independent of shuffling and of the shard assignment to workers/ranks,
so validation documents can never leak into training.
"""

import hashlib
import itertools
import os
from typing import Any, Callable, Iterator, List, Optional, Set, Tuple, Union
import warnings

import numpy as np
import torch
import torch.distributed as dist
from torch.utils.data import IterableDataset, get_worker_info

from dataset.in_memory_dataset import InMemoryDataset, TokenizerMixin

_TEXT_COLUMN_CANDIDATES = ("text", "content", "story", "raw_text", "sentence")


def _doc_hash(text: str) -> bytes:
    return hashlib.blake2b(text.encode("utf-8"), digest_size=16).digest()


def _dist_rank_world() -> Tuple[int, int]:
    """Rank and world size of the current process (torch.distributed, else torchrun env, else 0/1)."""
    if dist.is_available() and dist.is_initialized():
        return dist.get_rank(), dist.get_world_size()
    return int(os.environ.get("RANK", 0)), int(os.environ.get("WORLD_SIZE", 1))


class StreamingDataset(TokenizerMixin, IterableDataset):
    """
    Iterable PyTorch Dataset streaming documents and tokenizing them on the fly.

    Common usage:
        train_ds, val_ds = StreamingDataset.splits(
            "HuggingFaceFW/fineweb-edu", name="sample-10BT", seq_len=2048, val_num_docs=2000)

    Yields the same dicts as InMemoryDataset ('input_ids', 'attention_mask', 'labels'), each of
    length seq_len with labels[t] == input_ids[t+1]; consecutive sequences overlap by one token
    as in InMemoryDataset. Work is split across DDP ranks and DataLoader workers (by shards for
    HuggingFace datasets, round-robin over documents for plain iterables), so every document is
    seen by exactly one worker. There is no __len__: bound training with max_steps.
    """

    def __init__(
        self,
        dataset: Any,
        tokenizer: Any = "gpt2",
        seq_len: int = 1024,
        split: str = "train",
        name: Optional[str] = None,
        text_column: Optional[str] = None,
        special_token: Union[bool, str, int] = True,
        filter_fn: Optional[Callable[[str], bool]] = None,
        transform_fn: Optional[Callable[[str], str]] = None,
        exclude_hashes: Optional[Set[bytes]] = None,
        shuffle: bool = False,
        shuffle_buffer: int = 10_000,
        seq_shuffle_buffer: int = 0,
        seed: int = 0,
        batch_size: int = 512,
        num_threads: int = -1,
        verbose: bool = True,
    ):
        """
        Args:
            dataset: HuggingFace dataset name (str, loaded with streaming=True), a HuggingFace
                     (Iterable)Dataset, or any re-iterable collection of strings.
            tokenizer: Same options as InMemoryDataset. Default: 'gpt2'.
            seq_len: Sequence length of each yielded chunk.
            split: HuggingFace split to stream (used only when `dataset` is a name).
            name: HuggingFace dataset config name (e.g. 'sample-10BT' for FineWeb-Edu).
            text_column: Column containing the text (auto-detected if None).
            special_token: Delimiter appended after every document, as in InMemoryDataset.
            filter_fn / transform_fn: Same hooks as InMemoryDataset.
            exclude_hashes: Hashes (_doc_hash of the raw text) of documents to skip, e.g. the
                            validation documents. Filled automatically by splits().
            shuffle: Shuffle shard order and documents (HuggingFace buffer shuffle).
            shuffle_buffer: Document buffer size for the HuggingFace shuffle.
            seq_shuffle_buffer: If > 0, also shuffle the produced sequences through a buffer of
                                this size, so that the many sequences of a long document do not
                                end up next to each other in a batch.
            seed: Seed for both shuffles; call set_epoch() to reshuffle across epochs.
            batch_size: Number of documents tokenized together.
            num_threads: Tokenizer threads per DataLoader worker (-1: CPU count / workers).
            verbose: Print a summary at construction.
        """
        super().__init__()
        self.seq_len = int(seq_len)
        if self.seq_len < 1:
            raise ValueError(f"seq_len must be >= 1, got {self.seq_len}")
        self.split = self._normalize_split_name(split)
        self.name = name
        self.text_column = text_column
        self.filter_fn = filter_fn
        self.transform_fn = transform_fn
        self.exclude_hashes = exclude_hashes or set()
        self.shuffle = shuffle
        self.shuffle_buffer = shuffle_buffer
        self.seq_shuffle_buffer = seq_shuffle_buffer
        self.seed = seed
        self.epoch = 0
        self.batch_size = batch_size
        self._num_threads_arg = num_threads
        self.num_threads = os.cpu_count() if num_threads == -1 else num_threads
        self.verbose = verbose

        self.tokenizer, self.tokenizer_name, self.vocab_size = self._resolve_tokenizer(tokenizer)
        self.special_token_id = self._resolve_special_token(special_token)
        self.source = self._open_source(dataset)

        if self.verbose:
            print(
                f"[StreamingDataset] {self._describe(dataset)} split='{self.split}', seq_len={self.seq_len}, "
                f"tokenizer='{self.tokenizer_name}', shuffle={self.shuffle}, "
                f"excluded docs={len(self.exclude_hashes):,}"
            )

    # -------------------------------------------------------------------------
    # Source handling
    # -------------------------------------------------------------------------
    def _describe(self, dataset: Any) -> str:
        if isinstance(dataset, str):
            return f"'{dataset}'" + (f" ({self.name})" if self.name else "")
        return type(dataset).__name__

    def _open_source(self, dataset: Any) -> Any:
        if isinstance(dataset, str):
            import datasets

            dataset = datasets.load_dataset(dataset, self.name, split=self.split, streaming=True)
        if self._is_hf(dataset):
            if self.text_column is None:
                cols = dataset.column_names or []
                self.text_column = next((c for c in _TEXT_COLUMN_CANDIDATES if c in cols), None)
                if self.text_column is None:
                    raise ValueError(f"Could not determine text column from {cols}. Specify text_column='...'.")
            if not isinstance(dataset, _hf_iterable_type()):
                dataset = dataset.to_iterable_dataset()  # map-style HF Dataset
            if self.shuffle:
                # max_buffer_input_shards=1 keeps the shards separate (the default interleaves them
                # into a single shard), so DataLoader workers can still split them.
                dataset = dataset.shuffle(seed=self.seed, buffer_size=self.shuffle_buffer,
                                          max_buffer_input_shards=1)
        elif isinstance(dataset, Iterator):
            raise TypeError("Pass a re-iterable collection (list, HF dataset), not a one-shot iterator.")
        return dataset

    @staticmethod
    def _is_hf(dataset: Any) -> bool:
        return hasattr(dataset, "column_names") and hasattr(dataset, "features")

    def set_epoch(self, epoch: int):
        """Change the shuffling seed for a new pass over the data."""
        self.epoch = epoch
        if self._is_hf(self.source) and hasattr(self.source, "set_epoch"):
            self.source.set_epoch(epoch)

    def _iter_texts(self) -> Iterator[str]:
        """Raw texts assigned to this (rank, DataLoader worker)."""
        rank, world = _dist_rank_world()
        worker = get_worker_info()
        wid, nw = (worker.id, worker.num_workers) if worker is not None else (0, 1)

        if self._is_hf(self.source):
            import datasets.distributed

            src = self.source
            if world > 1:
                src = datasets.distributed.split_dataset_by_node(src, rank=rank, world_size=world)
            n_shards = getattr(src, "num_shards", 1)
            if worker is not None and n_shards < nw and wid == 0:
                warnings.warn(f"{nw} DataLoader workers but only {n_shards} shards: some workers stay idle.")
            # Inside a DataLoader worker, HF assigns shards to workers by itself.
            for ex in src:
                yield ex[self.text_column]
        else:
            gid, total = rank * nw + wid, world * nw
            for i, text in enumerate(self.source):
                if i % total == gid:
                    yield text

    # -------------------------------------------------------------------------
    # Tokenization and sequence packing
    # -------------------------------------------------------------------------
    def _iter_doc_batches(self) -> Iterator[List[str]]:
        batch: List[str] = []
        for text in self._iter_texts():
            if not text:
                continue
            if self.exclude_hashes and _doc_hash(text) in self.exclude_hashes:
                continue
            text = self._apply_transform(text)
            if not self._apply_filter(text):
                continue
            batch.append(text)
            if len(batch) == self.batch_size:
                yield batch
                batch = []
        if batch:
            yield batch

    def _iter_sequences(self) -> Iterator[np.ndarray]:
        """Chunks of seq_len + 1 tokens with stride seq_len (last token repeated as next first)."""
        L = self.seq_len
        buf = np.empty(0, dtype=np.int64)
        for texts in self._iter_doc_batches():
            docs = self._tokenize_batch(texts)
            if self.special_token_id is not None:
                docs = [d + [self.special_token_id] for d in docs]
            new = np.fromiter(itertools.chain.from_iterable(docs), dtype=np.int64)
            buf = np.concatenate([buf, new])
            n = (len(buf) - 1) // L
            for i in range(n):
                yield buf[i * L : i * L + L + 1]
            buf = buf[n * L :].copy()

    def _shuffled(self, seqs: Iterator[np.ndarray]) -> Iterator[np.ndarray]:
        rank, _ = _dist_rank_world()
        worker = get_worker_info()
        wid = worker.id if worker is not None else 0
        rng = np.random.default_rng([self.seed, self.epoch, rank, wid])
        pool: List[np.ndarray] = []
        for s in seqs:
            if len(pool) < self.seq_shuffle_buffer:
                pool.append(s)
                continue
            j = int(rng.integers(len(pool)))
            yield pool[j]
            pool[j] = s
        rng.shuffle(pool)
        yield from pool

    def __iter__(self):
        worker = get_worker_info()
        if self._num_threads_arg == -1 and worker is not None:
            self.num_threads = max(1, (os.cpu_count() or 1) // worker.num_workers)
        seqs = self._iter_sequences()
        if self.seq_shuffle_buffer > 0:
            seqs = self._shuffled(seqs)
        for s in seqs:
            t = torch.from_numpy(s)
            yield {
                "input_ids": t[:-1],
                "attention_mask": torch.ones(self.seq_len, dtype=torch.long),
                "labels": t[1:],
            }

    # -------------------------------------------------------------------------
    # Factory
    # -------------------------------------------------------------------------
    @classmethod
    def splits(
        cls,
        dataset: Any,
        tokenizer: Any = "gpt2",
        seq_len: int = 1024,
        name: Optional[str] = None,
        val_num_docs: int = 1000,
        text_column: Optional[str] = None,
        special_token: Union[bool, str, int] = True,
        filter_fn: Optional[Callable[[str], bool]] = None,
        transform_fn: Optional[Callable[[str], str]] = None,
        verbose: bool = True,
        **train_kwargs,
    ) -> Tuple["StreamingDataset", InMemoryDataset]:
        """
        Streaming train set and in-memory validation set.

        If `dataset` is a HuggingFace name with a 'validation' split, its first `val_num_docs`
        documents are the validation set; otherwise they are the first `val_num_docs`
        documents of 'train', which the training stream then excludes.
        `train_kwargs` go to the training StreamingDataset (shuffle, seed, buffers...).
        """
        common = dict(tokenizer=tokenizer, text_column=text_column, special_token=special_token,
                      filter_fn=filter_fn, transform_fn=transform_fn)
        has_val = False
        if isinstance(dataset, str):
            import datasets

            has_val = "validation" in datasets.get_dataset_split_names(dataset, name)
        val_stream = cls(dataset, seq_len=seq_len, split="validation" if has_val else "train",
                         name=name, verbose=False, **common)
        val_texts = []
        for text in val_stream.source:
            val_texts.append(text[val_stream.text_column] if val_stream._is_hf(val_stream.source) else text)
            if len(val_texts) == val_num_docs:
                break
        exclude = set() if has_val else {_doc_hash(t) for t in val_texts}
        val_ds = InMemoryDataset(val_texts, seq_len=seq_len, split="validation", cache=False,
                                 verbose=verbose, **common)
        train_ds = cls(dataset, seq_len=seq_len, split="train", name=name, exclude_hashes=exclude,
                       verbose=verbose, **common, **train_kwargs)
        return train_ds, val_ds


def _hf_iterable_type():
    import datasets

    return datasets.IterableDataset
