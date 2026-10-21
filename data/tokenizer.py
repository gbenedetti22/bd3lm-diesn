"""
GPT-2 BPE tokenizer (tiktoken) shared by the GPT and the refiner.

Token layout (identical to the previous DiESN project with the 'gpt2' backend):
    0 .. 50255   GPT-2 tokens
    50256        <|endoftext|>  document separator / BOS / EOS (the GPT decides the draft length with it)
    50257        [MASK]         refiner only (the GPT vocabulary is 50257, the refiner vocabulary 50258)
"""

import os

import tiktoken


class Tokenizer:
    def __init__(self, name: str = "gpt2"):
        self.name = name
        self.enc = tiktoken.get_encoding(name)
        self.vocab_size = self.enc.n_vocab  # GPT vocabulary (50257)
        self.eot_token_id = self.enc.eot_token  # 50256
        self.mask_token_id = self.vocab_size  # 50257, refiner only
        self.refiner_vocab_size = self.vocab_size + 1

    def encode(self, text: str) -> list[int]:
        return self.enc.encode_ordinary(text)

    def encode_batch(self, texts: list[str], num_threads: int = 0) -> list[list[int]]:
        return self.enc.encode_ordinary_batch(texts, num_threads=num_threads or min(32, os.cpu_count() or 1))

    def decode(self, ids, skip_eot: bool = True, mask_str: str = "") -> str:
        ids = [int(i) for i in ids]
        out = []
        for i in ids:
            if i == self.mask_token_id:
                if mask_str:
                    out.append(mask_str)
                continue
            if skip_eot and i == self.eot_token_id:
                continue
            out.append(i)
        if mask_str:  # decode piecewise to keep the mask placeholders
            text, buf = [], []
            for i in out:
                if isinstance(i, str):
                    text.append(self.enc.decode(buf))
                    buf = []
                    text.append(i)
                else:
                    buf.append(i)
            text.append(self.enc.decode(buf))
            return "".join(text)
        return self.enc.decode(out)

    def decode_batch(self, batch, skip_eot: bool = True) -> list[str]:
        return [self.decode(row.tolist() if hasattr(row, "tolist") else row, skip_eot=skip_eot) for row in batch]

    def word_tables(self):
        """
        Per-token lookup tables for whole-word masking (models.uncertainty.WordSegmenter), over the GPT vocabulary:
          attach (V,) int8: 2 = the token starts inside a UTF-8 character (continuation byte), so it always continues
                            the previous token; 1 = it starts with a letter or digit, so it continues the previous
                            token if that one ends with a letter or digit (" collect" + "ibles", " 17" + "th");
                            0 = it starts a word (leading space or newline, punctuation, <|endoftext|>)
          ends_in_word (V,) bool: the token ends with a letter or digit
        """
        import torch

        attach = torch.zeros(self.vocab_size, dtype=torch.int8)
        ends = torch.zeros(self.vocab_size, dtype=torch.bool)
        for t in range(self.vocab_size):
            b = self.enc.decode_single_token_bytes(t)
            text = b.decode("utf-8", errors="ignore")
            if b and (b[0] & 0xC0) == 0x80:
                attach[t] = 2
            elif text[:1].isalnum():
                attach[t] = 1
            ends[t] = text[-1:].isalnum()
        return attach, ends
