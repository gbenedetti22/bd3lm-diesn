"""
OpenWebText token streams of the provided dataset/ classes (dataset.OpenWebTextDataset: GPT-2 tokens, <|endoftext|>
after every document), with the last `val_docs` documents as validation and all the others as training.

The class caches every split in ~/.cache/gpr_datasets/<fingerprint>_tokens.npy (the fingerprint includes
val_num_docs). Tokenizing OpenWebText takes ~1.5 h: when the streams of another split are already cached (e.g. the
class default, validation = the last 1000 documents), the new streams are cut from them at the document boundary
instead. Documents are tokenized independently and only empty ones are dropped, so the result is identical to what
the class would build; it is saved under the names the class would use (it then finds it as its own cache).
"""

import os

import numpy as np

from data.token_datasets import EOT
from dataset.openwebtext import OpenWebTextDataset

HF_NAME = "Skylion007/openwebtext"


class _CachePath(OpenWebTextDataset):
    """OpenWebTextDataset that only resolves the path of its token cache (nothing is loaded or tokenized)."""

    def _load_or_build_tokens(self, dataset_arg):
        fingerprint = self._compute_cache_fingerprint(self._resolve_dataset_id(dataset_arg))
        self.cache_path = os.path.join(self.cache_dir, f"{fingerprint}_tokens.npy")
        return np.zeros(2, dtype=self.dtype)


def cache_paths(val_docs=None) -> dict:
    """{'train', 'val'} cache files of dataset.OpenWebTextDataset with the last `val_docs` documents as validation
    (None: the class default, 1000)."""
    return {key: _CachePath(split=split, seq_len=1, val_num_docs=val_docs, verbose=False).cache_path
            for key, split in (("train", "train"), ("val", "validation"))}


def _save(path, array):
    tmp = os.path.join(os.path.dirname(path), f".tmp_{os.path.basename(path)[:-4]}_{os.getpid()}.npy")
    np.save(tmp, array)  # atomic, as the class does
    os.replace(tmp, path)


def _derive(src: dict, dst: dict, src_val_docs: int, val_docs: int):
    """Move the validation boundary of the cached streams `src` (last `src_val_docs` documents) to the last
    `val_docs` documents (val_docs > src_val_docs) and save the new streams as `dst`."""
    import datasets
    import tiktoken

    docs = datasets.load_dataset(HF_NAME, split="train")
    n = len(docs)
    src_val_docs, val_docs = min(src_val_docs, n), min(val_docs, n)
    assert val_docs > src_val_docs, "the new validation split must contain the old one"
    keep = lambda t: bool(t) and bool(t.strip())  # noqa: E731  (InMemoryDataset._apply_filter)
    moved = docs.select(range(n - val_docs, n - src_val_docs))["text"]  # end of the old train stream
    m = sum(map(keep, moved))
    train = np.load(src["train"], mmap_mode="r")
    val = np.load(src["val"], mmap_mode="r")
    # the old train stream ends with the <|endoftext|> of its last document: cut after the (m + 1)-th one from the end
    tail = np.asarray(train[-(len(val) * (val_docs // src_val_docs + 2) + 10_000_000):])
    ends = np.flatnonzero(tail == EOT)
    assert len(ends) > m, "tail window too short"
    cut = len(train) - len(tail) + int(ends[-(m + 1)]) + 1
    new_val = np.concatenate([np.asarray(train[cut:]), np.asarray(val)])
    # checks: documents on both sides of the boundary, number of documents, total length
    enc = tiktoken.get_encoding("gpt2")
    first = next(t for t in moved if keep(t))
    last = next(docs[i]["text"] for i in range(n - val_docs - 1, -1, -1) if keep(docs[i]["text"]))
    a, b = enc.encode_ordinary(first), enc.encode_ordinary(last)
    assert new_val[:len(a)].tolist() == a and int(new_val[len(a)]) == EOT, "first validation document mismatch"
    assert np.asarray(train[cut - len(b) - 1:cut - 1]).tolist() == b and int(train[cut - 1]) == EOT, \
        "last training document mismatch"
    n_val_docs = sum(map(keep, docs.select(range(n - val_docs, n))["text"]))
    assert int((new_val == EOT).sum()) == n_val_docs, "validation document count mismatch"
    assert cut + len(new_val) == len(train) + len(val)
    _save(dst["val"], new_val)
    _save(dst["train"], train[:cut])
    return {"documents": n, "val_docs": n_val_docs, "train_tokens": cut, "val_tokens": len(new_val)}


def ensure_split(val_docs: int = 100_000, verbose: bool = True) -> dict:
    """Paths {'train', 'val'} of the streams with the last `val_docs` documents as validation, built if missing."""
    dst = cache_paths(val_docs)
    if all(os.path.exists(p) for p in dst.values()):
        return dst
    src = cache_paths(None)
    if all(os.path.exists(p) for p in src.values()) and val_docs > 1000:
        info = _derive(src, dst, 1000, val_docs)
        if verbose:
            print(f"[openwebtext] split from the cached default streams: {info}")
    else:  # tokenize with the provided class (~1.5 h)
        for split in ("train", "validation"):
            OpenWebTextDataset(split=split, val_num_docs=val_docs)
    return dst
