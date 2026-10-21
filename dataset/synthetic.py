"""
Synthetic long-context tasks for memory evaluation.

  - MQARDataset: multi-query associative recall. Key-value pairs, then `gap` filler tokens, then
    queries (a key followed by its value) at the end of the sequence.
  - PasskeyDataset: passkey retrieval inside repeated filler text (Mohtashami & Jaggi, 2023),
    with the needle placed `gap` tokens before the question.

Items use the InMemoryDataset format ({"input_ids", "attention_mask", "labels"}). Labels are -1
everywhere except where the next token is an answer, so the usual cross-entropy with
ignore_index=-1 trains and evaluates recall only. "distances" holds, for every labelled position,
how many tokens separate it from the fact it has to recall (-1 elsewhere).
Items are generated deterministically from (seed, index).
"""

from typing import Optional, Tuple

import numpy as np
import tiktoken
import torch
from torch.utils.data import Dataset


def _item(seq, answer_positions, distances):
    """seq: seq_len + 1 tokens. answer_positions: indices in seq of the tokens to predict."""
    seq = torch.as_tensor(seq, dtype=torch.long)
    x, y = seq[:-1], seq[1:].clone()
    keep = torch.zeros_like(y, dtype=torch.bool)
    dist = torch.full_like(y, -1)
    pos = torch.as_tensor(answer_positions, dtype=torch.long) - 1  # predicted at the previous position
    keep[pos] = True
    dist[pos] = torch.as_tensor(distances, dtype=torch.long)
    y[~keep] = -1
    return {"input_ids": x, "attention_mask": torch.ones_like(x), "labels": y, "distances": dist}


class MQARDataset(Dataset):
    """
    Multi-query associative recall (Zoology, Arora et al. 2023), with the facts far from the queries:

        layout="contiguous":  [filler]  k1 v1 ... kN vN  [gap filler tokens]  kq1 vq1 kq2 vq2 ... [end]
        layout="spread":      [filler k v filler .. k v filler]  [gap filler tokens]  kq1 vq1 ... [end]

    Keys, values and filler come from disjoint token ranges. The queried keys are a random subset
    of the pairs; the model must predict each vq after seeing kq. With "spread" the pairs are
    placed uniformly (one per bin) over everything before the gap. With a gap of at least the
    local window W, every fact is outside the window of every query.

    The defaults are calibrated so that a small full-attention model learns the task quickly (4
    layers x 256 reach 100% within a few thousand steps at seq_len 512 with a gap of 128): with
    1024 keys and few queries per sequence the same model needs far longer.
    """

    def __init__(
        self,
        num_examples: int,
        seq_len: int = 2048,
        num_pairs: int = 16,
        num_queries: int = 16,
        gap: Tuple[int, int] = (256, 1024),
        num_keys: int = 64,
        num_values: int = 64,
        key_start: int = 1000,
        value_start: int = 3000,
        filler_start: int = 5000,
        filler_vocab: int = 20000,
        layout: str = "spread",
        seed: int = 0,
    ):
        self.num_examples = num_examples
        self.seq_len = seq_len
        self.num_pairs = num_pairs
        self.num_queries = num_queries
        self.gap = (gap, gap) if isinstance(gap, int) else tuple(gap)
        self.num_keys, self.num_values = num_keys, num_values
        self.key_start, self.value_start = key_start, value_start
        self.filler_start, self.filler_vocab = filler_start, filler_vocab
        self.layout = layout
        self.seed = seed
        assert layout in ("contiguous", "spread"), f"Invalid layout: {layout}"
        assert num_queries <= num_pairs <= num_keys
        room = 2 * num_pairs * (2 if layout == "spread" else 1)
        assert room + 2 * num_queries + self.gap[1] <= seq_len + 1, "sequence too short"

    def __len__(self):
        return self.num_examples

    def __getitem__(self, index):
        rng = np.random.default_rng((self.seed, index))
        L = self.seq_len + 1
        seq = self.filler_start + rng.integers(0, self.filler_vocab, size=L)

        keys = self.key_start + rng.choice(self.num_keys, size=self.num_pairs, replace=False)
        values = self.value_start + rng.integers(0, self.num_values, size=self.num_pairs)
        gap = int(rng.integers(self.gap[0], self.gap[1] + 1))
        query_start = L - 2 * self.num_queries
        facts_end = query_start - gap  # every pair (key and value) is before this position
        if self.layout == "contiguous":
            pair_pos = facts_end - 2 * self.num_pairs + 2 * np.arange(self.num_pairs)
        else:
            width = facts_end // self.num_pairs
            pair_pos = width * np.arange(self.num_pairs) + rng.integers(0, width - 1, size=self.num_pairs)
        seq[pair_pos], seq[pair_pos + 1] = keys, values

        queried = rng.choice(self.num_pairs, size=self.num_queries, replace=False)
        query_pos = query_start + 2 * np.arange(self.num_queries)
        seq[query_pos] = keys[queried]
        seq[query_pos + 1] = values[queried]
        distances = query_pos - pair_pos[queried]
        return _item(seq, query_pos + 1, distances)


class PasskeyDataset(Dataset):
    """
    Passkey retrieval (Mohtashami & Jaggi, 2023): a random number hidden in repeated filler text,
    asked for at the end of the sequence. `gap` = filler tokens between the needle and the question.
    """

    PREFIX = ("There is an important info hidden inside a lot of irrelevant text. "
              "Find it and memorize it. I will quiz you about the important information there.")
    FILLER = " The grass is green. The sky is blue. The sun is yellow. Here we go. There and back again."
    NEEDLE = " The pass key is {key}. Remember it. {key} is the pass key."
    QUESTION = " What is the pass key? The pass key is"

    def __init__(
        self,
        num_examples: int,
        seq_len: int = 2048,
        gap: Tuple[int, int] = (256, 1024),
        digits: int = 5,
        tokenizer: Optional[tiktoken.Encoding] = None,
        seed: int = 0,
    ):
        self.num_examples = num_examples
        self.seq_len = seq_len
        self.gap = (gap, gap) if isinstance(gap, int) else tuple(gap)
        self.digits = digits
        self.seed = seed
        self.enc = tokenizer or tiktoken.get_encoding("gpt2")
        self.prefix = self.enc.encode_ordinary(self.PREFIX)
        self.filler = self.enc.encode_ordinary(self.FILLER)
        self.question = self.enc.encode_ordinary(self.QUESTION)

    def __len__(self):
        return self.num_examples

    def _fill(self, n):
        return (self.filler * (n // len(self.filler) + 1))[:n]

    def __getitem__(self, index):
        rng = np.random.default_rng((self.seed, index))
        key = str(int(rng.integers(10 ** (self.digits - 1), 10 ** self.digits)))
        needle = self.enc.encode_ordinary(self.NEEDLE.format(key=key))
        answer = self.enc.encode_ordinary(" " + key)
        gap = int(rng.integers(self.gap[0], self.gap[1] + 1))
        before = self.seq_len + 1 - len(self.prefix) - len(needle) - gap - len(self.question) - len(answer)
        assert before >= 0, "sequence too short for this gap"
        seq = self.prefix + self._fill(before) + needle + self._fill(gap) + self.question + answer
        answer_pos = np.arange(len(seq) - len(answer), len(seq))
        needle_pos = len(self.prefix) + before
        return _item(seq, answer_pos, answer_pos - needle_pos)
