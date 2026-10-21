"""
Tokenize a HuggingFace text dataset into flat uint16 token files (nanoGPT style).

    <data_dir>/<name>/train.bin, val.bin, meta.json

Every document is preceded by <|endoftext|> (document separator / BOS for the GPT).
"""

import json
import os
import time

import numpy as np

from data.tokenizer import Tokenizer

HF_DATASETS = {
    "tinystories": dict(path="roneneldan/TinyStories", config=None, train_split="train",
                        val_split="validation", text_field="text"),
    "fineweb-edu-10bt": dict(path="HuggingFaceFW/fineweb-edu", config="sample-10BT", train_split="train",
                             val_split=None, val_docs=100_000, text_field="text"),
    "wikitext103": dict(path="Salesforce/wikitext", config="wikitext-103-raw-v1", train_split="train",
                        val_split="validation", text_field="text"),
}


def _tokenize_split(dataset, field_: str, tok: Tokenizer, out_path: str, batch_docs=20_000):
    """Tokenize a HF Dataset split and write it to `out_path` (uint16). Returns the number of tokens."""
    n_docs = len(dataset)
    n_tokens = 0
    t0 = time.time()
    with open(out_path, "wb") as f:
        for i in range(0, n_docs, batch_docs):
            texts = [t for t in dataset[i:i + batch_docs][field_] if t]
            ids = tok.encode_batch(texts)
            lens = np.fromiter((len(d) + 1 for d in ids), dtype=np.int64, count=len(ids))
            arr = np.empty(int(lens.sum()), dtype=np.uint16)
            starts = np.concatenate([[0], np.cumsum(lens)[:-1]])
            for s, doc in zip(starts, ids):
                arr[s] = tok.eot_token_id
                arr[s + 1:s + 1 + len(doc)] = doc
            arr.tofile(f)
            n_tokens += len(arr)
            if (i // batch_docs) % 20 == 0:
                print(f"  {min(i + batch_docs, n_docs):,}/{n_docs:,} docs, {n_tokens:,} tokens "
                      f"({time.time() - t0:.0f}s)", flush=True)
    return n_tokens


def prepare_hf_dataset(name: str, data_dir: str = "data_cache", max_docs: int = 0):
    from datasets import load_dataset

    spec = HF_DATASETS[name]
    tok = Tokenizer()
    assert tok.refiner_vocab_size <= 65535, "uint16 storage requires vocab < 65536"
    root = os.path.join(data_dir, name)
    os.makedirs(root, exist_ok=True)
    ds = load_dataset(spec["path"], spec["config"]) if spec["config"] else load_dataset(spec["path"])
    train = ds[spec["train_split"]]
    if spec["val_split"]:
        val = ds[spec["val_split"]]
    else:  # last documents of the training split (MDLM/SMDM convention for OpenWebText)
        n, n_val = len(train), spec["val_docs"]
        val = train.select(range(n - n_val, n))
        train = train.select(range(n - n_val))
    if max_docs:
        train = train.select(range(min(max_docs, len(train))))
    meta = {"dataset": name, "hf_path": spec["path"], "hf_config": spec["config"], "tokenizer": tok.name,
            "dtype": "uint16", "eot_token_id": tok.eot_token_id}
    for split, dset in (("val", val), ("train", train)):
        path = os.path.join(root, f"{split}.bin")
        print(f"[prepare] {name}/{split}: tokenizing {len(dset):,} documents -> {path}")
        meta[f"{split}_tokens"] = _tokenize_split(dset, spec["text_field"], tok, path)
        meta[f"{split}_docs"] = len(dset)
    with open(os.path.join(root, "meta.json"), "w") as f:
        json.dump(meta, f, indent=2)
    print(f"[prepare] done: {meta}")
    return meta
