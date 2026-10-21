"""
Generic, high-performance in-memory dataset utility for training GPT models.
Supports multi-threaded batch tokenization, smart sequence-length independent caching,
zero-copy sequence chunking, train/val splitting, and custom filters.
"""

from abc import ABC
import functools
import hashlib
import inspect
import json
import os
import time
from typing import Any, Callable, Dict, List, Optional, Tuple, Union
import warnings

import numpy as np
import torch
from torch.utils.data import Dataset
import tiktoken
from tqdm import tqdm


@functools.lru_cache(maxsize=None)
def _hf_split_names(name: str) -> Tuple[str, ...]:
    """Split names of a HuggingFace dataset (one Hub request per name and process)."""
    import datasets

    return tuple(datasets.get_dataset_split_names(name))


def _get_default_cache_dir() -> str:
    """Return default cache directory ~/.cache/gpr_datasets."""
    return os.path.expanduser("~/.cache/gpr_datasets")


# =============================================================================
# Special Tokens Catalog & Helper Functions
# =============================================================================

_CANONICAL_SPECIAL_TOKENS = {
    # Beginning of sequence
    "bos": {
        "attr_ids": ["bos_token_id"],
        "attr_strs": ["bos_token"],
        "candidates": ["<|bos|>", "<s>", "<bos>", "[BOS]"],
    },
    "beginning_of_sequence": {
        "attr_ids": ["bos_token_id"],
        "attr_strs": ["bos_token"],
        "candidates": ["<|bos|>", "<s>", "<bos>", "[BOS]"],
    },
    "begging_of_sequence": {  # common spelling variant
        "attr_ids": ["bos_token_id"],
        "attr_strs": ["bos_token"],
        "candidates": ["<|bos|>", "<s>", "<bos>", "[BOS]"],
    },
    # End of sequence
    "eos": {
        "attr_ids": ["eos_token_id"],
        "attr_strs": ["eos_token"],
        "candidates": ["<|eos|>", "</s>", "<eos>", "[EOS]"],
    },
    "end_of_sequence": {
        "attr_ids": ["eos_token_id"],
        "attr_strs": ["eos_token"],
        "candidates": ["<|eos|>", "</s>", "<eos>", "[EOS]"],
    },
    # Start of text
    "start_of_text": {
        "attr_ids": [],
        "attr_strs": [],
        "candidates": ["<|startoftext|>", "<|start_of_text|>", "<|sot|>", "[STARTOFTEXT]"],
        "fallback_to_eot": True,  # in tiktoken/GPT models, <|endoftext|> is used as document delimiter
    },
    "startoftext": {
        "attr_ids": [],
        "attr_strs": [],
        "candidates": ["<|startoftext|>", "<|start_of_text|>", "<|sot|>", "[STARTOFTEXT]"],
        "fallback_to_eot": True,
    },
    # End of text
    "end_of_text": {
        "attr_ids": ["eot_token"],
        "attr_strs": [],
        "candidates": ["<|endoftext|>", "<|end_of_text|>", "<|eot|>", "[ENDOFTEXT]"],
    },
    "endoftext": {
        "attr_ids": ["eot_token"],
        "attr_strs": [],
        "candidates": ["<|endoftext|>", "<|end_of_text|>", "<|eot|>", "[ENDOFTEXT]"],
    },
    # Pad
    "pad": {
        "attr_ids": ["pad_token_id"],
        "attr_strs": ["pad_token"],
        "candidates": ["<|pad|>", "<pad>", "[PAD]"],
    },
    "pad_token": {
        "attr_ids": ["pad_token_id"],
        "attr_strs": ["pad_token"],
        "candidates": ["<|pad|>", "<pad>", "[PAD]"],
    },
    # UNK
    "unk": {
        "attr_ids": ["unk_token_id"],
        "attr_strs": ["unk_token"],
        "candidates": ["<|unk|>", "<unk>", "[UNK]"],
    },
    "unknown": {
        "attr_ids": ["unk_token_id"],
        "attr_strs": ["unk_token"],
        "candidates": ["<|unk|>", "<unk>", "[UNK]"],
    },
    # SEP
    "sep": {
        "attr_ids": ["sep_token_id"],
        "attr_strs": ["sep_token"],
        "candidates": ["<|sep|>", "<sep>", "[SEP]"],
    },
    "separator": {
        "attr_ids": ["sep_token_id"],
        "attr_strs": ["sep_token"],
        "candidates": ["<|sep|>", "<sep>", "[SEP]"],
    },
    # MASK
    "mask": {
        "attr_ids": ["mask_token_id"],
        "attr_strs": ["mask_token"],
        "candidates": ["<|mask|>", "<mask>", "[MASK]"],
    },
    # CLS
    "cls": {
        "attr_ids": ["cls_token_id"],
        "attr_strs": ["cls_token"],
        "candidates": ["<|cls|>", "<cls>", "[CLS]"],
    },
}


def get_available_special_tokens(tokenizer: Any) -> Dict[str, int]:
    """
    Return a dictionary mapping all available special token strings to their integer IDs.
    Works across tiktoken, HuggingFace, RustBPETokenizer, and custom tokenizers.
    """
    specials: Dict[str, int] = {}

    # 1. tiktoken
    if hasattr(tokenizer, "_special_tokens") and isinstance(tokenizer._special_tokens, dict):
        specials.update(tokenizer._special_tokens)
    elif hasattr(tokenizer, "special_tokens_set") and hasattr(tokenizer, "encode_single_token"):
        for tok_str in tokenizer.special_tokens_set:
            try:
                specials[tok_str] = tokenizer.encode_single_token(tok_str)
            except Exception:
                pass

    if hasattr(tokenizer, "eot_token") and tokenizer.eot_token is not None:
        specials.setdefault("<|endoftext|>", tokenizer.eot_token)

    # 2. Hugging Face
    if hasattr(tokenizer, "all_special_tokens") and hasattr(tokenizer, "all_special_ids"):
        for tok_str, tok_id in zip(tokenizer.all_special_tokens, tokenizer.all_special_ids):
            specials[tok_str] = tok_id
    elif hasattr(tokenizer, "special_tokens_map"):
        for key, val in tokenizer.special_tokens_map.items():
            if isinstance(val, str) and hasattr(tokenizer, "convert_tokens_to_ids"):
                tok_id = tokenizer.convert_tokens_to_ids(val)
                if tok_id is not None:
                    specials[val] = tok_id

    # 3. Nanochat RustBPETokenizer / custom
    if hasattr(tokenizer, "bos_token_id") and tokenizer.bos_token_id is not None:
        specials.setdefault("<|bos|>", tokenizer.bos_token_id)

    return specials


def get_special_token_id(tokenizer: Any, name: str) -> int:
    """
    Retrieve the integer token ID for a special token.

    Args:
        tokenizer: tiktoken Encoding, HuggingFace Tokenizer, or tokenizer object.
        name: Name of special token (e.g. 'bos', 'eos', 'start_of_text', 'end_of_text',
              'pad', 'unk', 'sep', 'mask', or exact string like '<|endoftext|>').

    Returns:
        int: Token ID.

    Raises:
        ValueError: If the special token is not present or supported in the tokenizer.
    """
    raw_name = name.strip()
    norm_name = raw_name.lower().replace("-", "_").replace(" ", "_")

    # 1. Direct lookup in tokenizer if name is an exact token string (e.g. '<|endoftext|>')
    if (raw_name.startswith("<") and raw_name.endswith(">")) or (raw_name.startswith("[") and raw_name.endswith("]")):
        if hasattr(tokenizer, "_special_tokens") and raw_name in tokenizer._special_tokens:
            return tokenizer._special_tokens[raw_name]
        if hasattr(tokenizer, "encode_single_token"):
            try:
                return tokenizer.encode_single_token(raw_name)
            except (KeyError, ValueError):
                pass
        if hasattr(tokenizer, "convert_tokens_to_ids"):
            tid = tokenizer.convert_tokens_to_ids(raw_name)
            unk_id = getattr(tokenizer, "unk_token_id", -1)
            if tid is not None and (tid != unk_id or raw_name == getattr(tokenizer, "unk_token", None)):
                return tid

    # 2. Check canonical mapping
    entry = _CANONICAL_SPECIAL_TOKENS.get(norm_name)
    if entry is not None:
        # 2a. Check direct tokenizer attribute IDs (e.g. bos_token_id, eos_token_id, eot_token)
        for attr in entry["attr_ids"]:
            if hasattr(tokenizer, attr):
                val = getattr(tokenizer, attr)
                if val is not None and isinstance(val, int):
                    return val

        # 2b. Check direct tokenizer attribute string (e.g. bos_token -> convert to ID)
        for attr in entry["attr_strs"]:
            if hasattr(tokenizer, attr):
                val = getattr(tokenizer, attr)
                if val is not None and isinstance(val, str):
                    if hasattr(tokenizer, "convert_tokens_to_ids"):
                        tid = tokenizer.convert_tokens_to_ids(val)
                        if tid is not None:
                            return tid
                    if hasattr(tokenizer, "encode_single_token"):
                        try:
                            return tokenizer.encode_single_token(val)
                        except (KeyError, ValueError):
                            pass

        # 2c. Check candidate literal strings
        for cand in entry["candidates"]:
            if hasattr(tokenizer, "_special_tokens") and cand in tokenizer._special_tokens:
                return tokenizer._special_tokens[cand]
            if hasattr(tokenizer, "encode_single_token"):
                try:
                    return tokenizer.encode_single_token(cand)
                except (KeyError, ValueError):
                    pass
            if hasattr(tokenizer, "convert_tokens_to_ids"):
                tid = tokenizer.convert_tokens_to_ids(cand)
                unk_id = getattr(tokenizer, "unk_token_id", -1)
                if tid is not None and (tid != unk_id or cand == getattr(tokenizer, "unk_token", None)):
                    return tid

        # 2d. Fallback for start_of_text in tiktoken/GPT models:
        # In OpenAI GPT models, <|endoftext|> acts as document delimiter (start/end of text).
        if entry.get("fallback_to_eot", False):
            if hasattr(tokenizer, "eot_token") and tokenizer.eot_token is not None:
                return tokenizer.eot_token
            if hasattr(tokenizer, "_special_tokens") and "<|endoftext|>" in tokenizer._special_tokens:
                return tokenizer._special_tokens["<|endoftext|>"]

    # 3. Try exact string in available special tokens
    avail = get_available_special_tokens(tokenizer)
    if raw_name in avail:
        return avail[raw_name]

    tok_name = getattr(tokenizer, "name", type(tokenizer).__name__)
    avail_keys = list(avail.keys())
    raise ValueError(
        f"Special token '{name}' is not present or supported in tokenizer '{tok_name}'. "
        f"Available special tokens in this tokenizer: {avail_keys}"
    )


def get_special_token_str(tokenizer: Any, name: str) -> str:
    """
    Retrieve the string representation of a special token (e.g. '<|endoftext|>').

    Raises:
        ValueError: If the special token is not present or supported in the tokenizer.
    """
    tok_id = get_special_token_id(tokenizer, name)
    if hasattr(tokenizer, "decode_single_token_bytes"):
        return tokenizer.decode_single_token_bytes(tok_id).decode("utf-8", errors="replace")
    if hasattr(tokenizer, "decode"):
        return tokenizer.decode([tok_id])
    if hasattr(tokenizer, "convert_ids_to_tokens"):
        return str(tokenizer.convert_ids_to_tokens(tok_id))

    tok_name = getattr(tokenizer, "name", type(tokenizer).__name__)
    raise ValueError(f"Cannot decode token ID {tok_id} to string for tokenizer '{tok_name}'.")


def get_bos_token_id(tokenizer: Any) -> int:
    """Get Beginning of Sequence (BOS) token ID or raise ValueError."""
    return get_special_token_id(tokenizer, "bos")


def get_eos_token_id(tokenizer: Any) -> int:
    """Get End of Sequence (EOS) token ID or raise ValueError."""
    return get_special_token_id(tokenizer, "eos")


def get_start_of_text_token_id(tokenizer: Any) -> int:
    """Get Start of Text token ID or raise ValueError."""
    return get_special_token_id(tokenizer, "start_of_text")


def get_end_of_text_token_id(tokenizer: Any) -> int:
    """Get End of Text token ID or raise ValueError."""
    return get_special_token_id(tokenizer, "end_of_text")


def get_pad_token_id(tokenizer: Any) -> int:
    """Get PAD token ID or raise ValueError."""
    return get_special_token_id(tokenizer, "pad")


def get_unk_token_id(tokenizer: Any) -> int:
    """Get UNK token ID or raise ValueError."""
    return get_special_token_id(tokenizer, "unk")


def get_sep_token_id(tokenizer: Any) -> int:
    """Get SEP token ID or raise ValueError."""
    return get_special_token_id(tokenizer, "sep")


def get_mask_token_id(tokenizer: Any) -> int:
    """Get MASK token ID or raise ValueError."""
    return get_special_token_id(tokenizer, "mask")


def get_cls_token_id(tokenizer: Any) -> int:
    """Get CLS token ID or raise ValueError."""
    return get_special_token_id(tokenizer, "cls")


class TokenizerMixin:
    """
    Tokenizer, special-token and text-hook logic shared by InMemoryDataset and StreamingDataset.
    Expects the attributes tokenizer, filter_fn, transform_fn and num_threads on the instance.
    """

    # -------------------------------------------------------------------------
    # Split Name Normalization
    # -------------------------------------------------------------------------
    @staticmethod
    def _normalize_split_name(split: str) -> str:
        s = split.lower().strip()
        if s in ("val", "valid", "validation"):
            return "validation"
        if s in ("train", "training"):
            return "train"
        if s in ("test", "testing"):
            return "test"
        return s

    # -------------------------------------------------------------------------
    # Tokenizer & Special Token Resolution
    # -------------------------------------------------------------------------
    def _resolve_tokenizer(self, tokenizer: Any) -> Tuple[Any, str, Optional[int]]:
        """Resolve tokenizer instance, identifier name, and vocab size."""
        if tokenizer is None or (isinstance(tokenizer, str) and tokenizer == "gpt2"):
            enc = tiktoken.get_encoding("gpt2")
            return enc, "tiktoken:gpt2", enc.n_vocab

        if isinstance(tokenizer, str):
            try:
                enc = tiktoken.get_encoding(tokenizer)
                return enc, f"tiktoken:{tokenizer}", enc.n_vocab
            except ValueError:
                pass
            try:
                enc = tiktoken.encoding_for_model(tokenizer)
                return enc, f"tiktoken_model:{tokenizer}", enc.n_vocab
            except KeyError:
                pass

            # Optional fallback to HuggingFace Transformers if installed
            import importlib
            if importlib.util.find_spec("transformers") is not None:
                try:
                    transformers = importlib.import_module("transformers")
                    hf_tok = transformers.AutoTokenizer.from_pretrained(tokenizer)
                    return hf_tok, f"hf:{tokenizer}", getattr(hf_tok, "vocab_size", None)
                except Exception as e:
                    raise ValueError(f"Could not load HuggingFace tokenizer '{tokenizer}': {e}")
            raise ValueError(f"Unknown tiktoken encoding/model '{tokenizer}' and transformers is not installed.")

        # Object or custom tokenizer
        if hasattr(tokenizer, "n_vocab"):  # tiktoken Encoding
            return tokenizer, getattr(tokenizer, "name", "tiktoken_custom"), tokenizer.n_vocab
        if hasattr(tokenizer, "get_vocab_size"):  # Nanochat RustBPETokenizer
            return tokenizer, "rustbpe", tokenizer.get_vocab_size()
        if hasattr(tokenizer, "vocab_size"):  # HF PreTrainedTokenizer
            name = getattr(tokenizer, "name_or_path", "hf_tokenizer")
            return tokenizer, f"hf:{name}", tokenizer.vocab_size
        if callable(tokenizer):
            name = getattr(tokenizer, "__name__", "custom_callable")
            return tokenizer, f"callable:{name}", None

        raise TypeError(f"Unsupported tokenizer type: {type(tokenizer)}")

    def _resolve_special_token(self, special_token: Union[bool, str, int]) -> Optional[int]:
        """Resolve special delimiter token ID to insert between documents."""
        if special_token is False or special_token is None:
            return None

        # bool is a subclass of int: True must go to auto-detection, not become token id 1
        if isinstance(special_token, int) and not isinstance(special_token, bool):
            return special_token

        tok = self.tokenizer

        if isinstance(special_token, str):
            if hasattr(tok, "encode_single_token"):
                return tok.encode_single_token(special_token)
            if hasattr(tok, "encode_special"):
                return tok.encode_special(special_token)
            if hasattr(tok, "convert_tokens_to_ids"):
                return tok.convert_tokens_to_ids(special_token)
            encoded = self._tokenize_single(special_token)
            if len(encoded) == 1:
                return encoded[0]
            raise ValueError(f"Special token string '{special_token}' encoded to multiple tokens: {encoded}")

        # special_token is True: Auto-detect
        if hasattr(tok, "eot_token") and tok.eot_token is not None:
            return tok.eot_token
        if hasattr(tok, "bos_token_id") and tok.bos_token_id is not None:
            return tok.bos_token_id
        if hasattr(tok, "eos_token_id") and tok.eos_token_id is not None:
            return tok.eos_token_id
        if hasattr(tok, "encode_single_token"):
            for candidate in ("<|endoftext|>", "<|bos|>"):
                try:
                    return tok.encode_single_token(candidate)
                except KeyError:
                    pass

        warnings.warn(
            "Could not auto-detect special token (BOS/EOS/EOT). No delimiter token will be inserted."
        )
        return None

    # -------------------------------------------------------------------------
    # Tokenization Methods
    # -------------------------------------------------------------------------
    def _tokenize_single(self, text: str) -> List[int]:
        """Tokenize a single text string."""
        # Check subclass override
        if hasattr(self, "_tokenize") and callable(self._tokenize):
            return self._tokenize(text)

        tok = self.tokenizer
        if hasattr(tok, "encode_ordinary"):
            return tok.encode_ordinary(text)
        if hasattr(tok, "encode"):
            return tok.encode(text)
        if callable(tok):
            return tok(text)
        raise RuntimeError(f"Tokenizer {type(tok)} has no standard encode method.")

    def _tokenize_batch(self, texts: List[str]) -> List[List[int]]:
        """Fast multi-threaded batch tokenization."""
        # Check subclass override
        if hasattr(self, "tokenize_batch") and callable(self.tokenize_batch):
            return self.tokenize_batch(texts)

        tok = self.tokenizer
        if hasattr(tok, "encode_ordinary_batch"):
            return tok.encode_ordinary_batch(texts, num_threads=self.num_threads)
        if hasattr(tok, "encode_batch"):
            return [e.ids for e in tok.encode_batch(texts)]
        if hasattr(tok, "__call__") and hasattr(tok, "is_fast") and tok.is_fast:
            out = tok(texts, add_special_tokens=False, return_attention_mask=False)
            return out["input_ids"]

        # Fallback to single tokenize loop
        return [self._tokenize_single(t) for t in texts]

    # -------------------------------------------------------------------------
    # Filtering and Preprocessing Hooks
    # -------------------------------------------------------------------------
    def _apply_filter(self, text: str) -> bool:
        """Return True if text passes filtering, False to discard."""
        if not text or len(text.strip()) == 0:
            return False

        if self.filter_fn is not None:
            if not self.filter_fn(text):
                return False

        # Subclass hook: support _filter or __filter__
        if hasattr(self, "_filter") and callable(self._filter):
            return self._filter(text)
        if hasattr(self, "__filter__") and callable(self.__filter__):
            res = self.__filter__(text)
            if res is not None:
                return bool(res)

        return True

    def _apply_transform(self, text: str) -> str:
        """Apply optional text transformation/cleaning before tokenization."""
        if self.transform_fn is not None:
            return self.transform_fn(text)
        return text

    # -------------------------------------------------------------------------
    # Special Token Accessors & Helpers
    # -------------------------------------------------------------------------
    def get_special_token(self, name: str) -> int:
        """
        Get the integer ID of a special token by name or string.
        Raises ValueError if not present in this dataset's tokenizer.
        """
        return get_special_token_id(self.tokenizer, name)

    def get_special_token_string(self, name: str) -> str:
        """
        Get the decoded string of a special token (e.g. '<|endoftext|>').
        Raises ValueError if not present in this dataset's tokenizer.
        """
        return get_special_token_str(self.tokenizer, name)

    @property
    def special_tokens(self) -> Dict[str, int]:
        """Dictionary mapping all available special token strings to integer IDs."""
        return get_available_special_tokens(self.tokenizer)

    @property
    def bos_token_id(self) -> int:
        """Beginning of Sequence (BOS) token ID. Raises ValueError if not present."""
        return get_bos_token_id(self.tokenizer)

    @property
    def eos_token_id(self) -> int:
        """End of Sequence (EOS) token ID. Raises ValueError if not present."""
        return get_eos_token_id(self.tokenizer)

    @property
    def start_of_text_token_id(self) -> int:
        """Start of text token ID. Raises ValueError if not present."""
        return get_start_of_text_token_id(self.tokenizer)

    @property
    def start_of_text_id(self) -> int:
        """Alias for start_of_text_token_id."""
        return self.start_of_text_token_id

    @property
    def end_of_text_token_id(self) -> int:
        """End of text token ID. Raises ValueError if not present."""
        return get_end_of_text_token_id(self.tokenizer)

    @property
    def end_of_text_id(self) -> int:
        """Alias for end_of_text_token_id."""
        return self.end_of_text_token_id

    @property
    def pad_token_id(self) -> int:
        """PAD token ID. Raises ValueError if not present."""
        return get_pad_token_id(self.tokenizer)

    @property
    def unk_token_id(self) -> int:
        """UNK token ID. Raises ValueError if not present."""
        return get_unk_token_id(self.tokenizer)

    @property
    def sep_token_id(self) -> int:
        """SEP token ID. Raises ValueError if not present."""
        return get_sep_token_id(self.tokenizer)

    @property
    def mask_token_id(self) -> int:
        """MASK token ID. Raises ValueError if not present."""
        return get_mask_token_id(self.tokenizer)

    @property
    def cls_token_id(self) -> int:
        """CLS token ID. Raises ValueError if not present."""
        return get_cls_token_id(self.tokenizer)



class InMemoryDataset(TokenizerMixin, Dataset, ABC):
    """
    High-performance in-memory PyTorch Dataset for GPT autoregressive training.

    Common usage:
        dataset = InMemoryDataset("roneneldan/TinyStories", seq_len=1024, split="train")

    Features:
        - Defaults to tiktoken ('gpt2') for zero-friction setup.
        - Loads tokens entirely into RAM for ultra-fast, zero-overhead training batches.
        - Sequence-length independent caching: caches the entire tokenized dataset once;
          experimenting with different sequence lengths requires NO re-tokenization.
        - Fast multi-threaded batch tokenization (millions of tokens/sec).
        - Zero-copy strided chunking yielding (x, y) pairs where y_t = x_{t+1}.
        - Automatic or customizable train/validation split (including last N docs).
        - Special token insertion between documents (enabled by default).
        - Document filtering and transformation hooks.
    """

    def __init__(
        self,
        dataset: Any = None,
        tokenizer: Any = "gpt2",
        seq_len: int = 1024,
        split: str = "train",
        text_column: Optional[str] = None,
        val_num_docs: Optional[int] = None,
        val_ratio: Optional[float] = None,
        special_token: Union[bool, str, int] = True,
        filter_fn: Optional[Callable[[str], bool]] = None,
        transform_fn: Optional[Callable[[str], str]] = None,
        cache: bool = True,
        cache_dir: Optional[str] = None,
        force_rebuild: bool = False,
        dtype: Optional[np.dtype] = None,
        batch_size: int = 4096,
        num_threads: int = -1,
        verbose: bool = True,
    ):
        """
        Args:
            dataset: HuggingFace dataset name (str), HuggingFace Dataset/DatasetDict,
                     iterable of strings, or None (if subclass overrides _load_dataset).
            tokenizer: tiktoken encoding name (e.g. 'gpt2'), tiktoken Encoding object,
                       HuggingFace tokenizer, custom callable, or None. Default: 'gpt2'.
            seq_len: Sequence length for autoregressive language model input chunks.
            split: 'train' or 'val' / 'validation'.
            text_column: Column name containing text in dataset (auto-detected if None).
            val_num_docs: If dataset has no validation split, use the last N documents
                          as validation set (and remaining for train).
            val_ratio: Fraction of documents to use for validation if val_num_docs not set.
            special_token: Delimiter token between documents. True (auto-detect BOS/EOS/EOT),
                           str/int for explicit token, or False to disable. Default: True.
            filter_fn: Callable(text: str) -> bool. If returns False, document is dropped.
            transform_fn: Callable(text: str) -> str. Optional preprocessing applied before tokenizing.
            cache: Whether to cache the tokenized dataset to disk. Default: True.
            cache_dir: Directory to store cached tokens. Defaults to ~/.cache/gpr_datasets.
            force_rebuild: If True, ignore existing cache and rebuild token stream.
            dtype: Numpy integer dtype for token storage (np.uint16 or np.int32).
                   If None, auto-selects np.uint16 if vocab_size <= 65535, else np.int32.
            batch_size: Number of documents to batch tokenize at once.
            num_threads: Number of threads used during batch tokenization.
            verbose: If True, prints informative progress and caching logs.
        """
        super().__init__()
        self.seq_len = int(seq_len)
        if self.seq_len < 1:
            raise ValueError(f"seq_len must be >= 1, got {self.seq_len}")

        self.split = self._normalize_split_name(split)
        self.text_column = text_column
        self.val_num_docs = val_num_docs
        self.val_ratio = val_ratio
        self.filter_fn = filter_fn
        self.transform_fn = transform_fn
        self.cache = cache
        self.cache_dir = cache_dir or _get_default_cache_dir()
        self.force_rebuild = force_rebuild
        self.batch_size = batch_size
        self.num_threads = os.cpu_count() if num_threads == -1 else num_threads
        self.verbose = verbose

        # 1. Resolve tokenizer and special token
        self.tokenizer, self.tokenizer_name, self.vocab_size = self._resolve_tokenizer(tokenizer)
        self.special_token_id = self._resolve_special_token(special_token)

        # 2. Select storage dtype
        if dtype is not None:
            self.dtype = np.dtype(dtype)
        else:
            # Auto-select compact dtype (uint16 saves 2x RAM over int32 and 4x over int64)
            max_id = self.vocab_size if self.vocab_size is not None else 65535
            if self.special_token_id is not None:
                max_id = max(max_id, self.special_token_id)
            self.dtype = np.uint16 if max_id <= 65535 else np.int32

        # 3. Load or prepare token stream (Sequence-Length Independent!)
        self.tokens_np = self._load_or_build_tokens(dataset)

        # 4. Total tokens and sequence calculations
        self.total_tokens = len(self.tokens_np)
        self.num_sequences = max(0, (self.total_tokens - 1) // self.seq_len)
        if self.num_sequences == 0:
            raise ValueError(
                f"Dataset contains {self.total_tokens} tokens, which is fewer than "
                f"seq_len + 1 ({self.seq_len + 1}). Cannot form any full sequence."
            )

        # Keep tokens entirely in RAM
        # Store as torch tensor view sharing the same numpy buffer (zero copy!)
        self.tokens = torch.from_numpy(self.tokens_np)

        if self.verbose:
            print(
                f"[InMemoryDataset] Loaded split='{self.split}' in memory: "
                f"{self.total_tokens:,} tokens ({self.tokens_np.nbytes / (1024*1024):.2f} MB), "
                f"{self.num_sequences:,} sequences of len={self.seq_len}."
            )

    # -------------------------------------------------------------------------
    # Raw Data Loading & Slicing
    # -------------------------------------------------------------------------
    def _load_raw_data(self, dataset_arg: Any) -> Tuple[List[str], str]:
        """
        Load raw text documents for the requested split.
        Returns:
            texts: List[str] of documents
            dataset_id: Identifier string for caching
        """
        # If dataset_arg not provided, check subclass _load_dataset or __load_dataset__
        if dataset_arg is None:
            if hasattr(self, "_load_dataset") and callable(self._load_dataset):
                dataset_arg = self._load_dataset()
            elif hasattr(self, "__load_dataset__") and callable(self.__load_dataset__):
                dataset_arg = self.__load_dataset__()

        if dataset_arg is None:
            raise ValueError("No dataset provided and subclass does not implement _load_dataset().")

        # Case 1: Dict with {'ds': ..., 'train_key': ..., 'val_key': ...} (user draft style)
        if isinstance(dataset_arg, dict) and "ds" in dataset_arg:
            ds = dataset_arg["ds"]
            train_key = dataset_arg.get("train_key", "train")
            val_key = dataset_arg.get("val_key", "validation")
            key = train_key if self.split == "train" else val_key
            target_ds = ds[key]
            dataset_id = f"dict_ds:{train_key}_{val_key}"
            texts = self._extract_texts(target_ds)
            return texts, dataset_id

        # Case 2: HuggingFace dataset name string (e.g. "roneneldan/TinyStories")
        if isinstance(dataset_arg, str):
            dataset_id = dataset_arg
            import datasets

            # Custom document splitting: requested (val_num_docs / val_ratio), or needed because the
            # dataset has no validation split (then train must exclude the validation documents)
            if self._needs_partition(dataset_arg):
                # Load train split (or full dataset) and perform custom split
                full_ds = datasets.load_dataset(dataset_arg, split="train")
                texts = self._extract_texts(full_ds)
                texts = self._partition_texts(texts)
                return texts, f"{dataset_id}:partitioned_{self.split}"

            # Standard split loading
            try:
                hf_split = "validation" if self.split == "validation" else self.split
                ds = datasets.load_dataset(dataset_arg, split=hf_split)
                texts = self._extract_texts(ds)
                return texts, dataset_id
            except Exception:
                # If specific split not found, attempt loading entire dataset and partition
                full_ds = datasets.load_dataset(dataset_arg)
                if isinstance(full_ds, dict):
                    # Check available keys
                    if self.split in full_ds:
                        texts = self._extract_texts(full_ds[self.split])
                        return texts, dataset_id
                    elif "train" in full_ds:
                        texts = self._extract_texts(full_ds["train"])
                        texts = self._partition_texts(texts)
                        return texts, f"{dataset_id}:auto_split_{self.split}"
                raise

        # Case 3: HuggingFace Dataset / DatasetDict object
        if hasattr(dataset_arg, "column_names") or isinstance(dataset_arg, dict):
            if isinstance(dataset_arg, dict) and self.split in dataset_arg:
                target_ds = dataset_arg[self.split]
                dataset_id = f"hf_object:{self.split}"
                texts = self._extract_texts(target_ds)
                return texts, dataset_id
            texts = self._extract_texts(dataset_arg)
            if self.val_num_docs is not None or self.val_ratio is not None:
                texts = self._partition_texts(texts)
                return texts, f"custom_object:partitioned_{self.split}"
            return texts, "custom_object"

        # Case 4: In-memory list/iterable of strings
        if isinstance(dataset_arg, (list, tuple)):
            texts = list(dataset_arg)
            if self.val_num_docs is not None or self.val_ratio is not None:
                texts = self._partition_texts(texts)
            return texts, f"list_data:len_{len(dataset_arg)}"

        raise TypeError(f"Unsupported dataset format: {type(dataset_arg)}")

    def _needs_partition(self, name: str) -> bool:
        """Whether train/validation come from partitioning the 'train' split of HF dataset `name`."""
        if self.val_num_docs is not None or self.val_ratio is not None:
            return True
        if self.split not in ("train", "validation"):
            return False
        return "validation" not in _hf_split_names(name)

    def _extract_texts(self, ds: Any) -> List[str]:
        """Extract list of text strings from a HuggingFace Dataset or list."""
        if isinstance(ds, (list, tuple)):
            return list(ds)

        if hasattr(ds, "column_names"):
            col = self.text_column
            if col is None:
                for candidate in ("text", "content", "story", "raw_text", "sentence"):
                    if candidate in ds.column_names:
                        col = candidate
                        break
            if col is None or col not in ds.column_names:
                raise ValueError(
                    f"Could not determine text column from {ds.column_names}. Specify text_column='...'."
                )
            return ds[col]

        if isinstance(ds, dict):
            for k in ("text", "content", "story"):
                if k in ds:
                    return ds[k]

        return list(ds)

    def _partition_texts(self, texts: List[str]) -> List[str]:
        """Partition texts when dataset does not provide validation split."""
        total = len(texts)
        if self.val_num_docs is not None:
            n_val = min(self.val_num_docs, total)
        elif self.val_ratio is not None:
            n_val = int(total * self.val_ratio)
        else:
            n_val = min(1000, total // 10)

        if self.split == "validation":
            return texts[-n_val:]
        return texts[:-n_val]

    # -------------------------------------------------------------------------
    # Caching Mechanism (Sequence-Length Independent!)
    # -------------------------------------------------------------------------
    def _compute_cache_fingerprint(self, dataset_id: str) -> str:
        """
        Compute SHA-256 cache key based on data identity and tokenization parameters.
        Notice: seq_len is deliberately EXCLUDED so the cache can be reused across any seq_len!
        """
        components = {
            "dataset_id": dataset_id,
            "split": self.split,
            "tokenizer_name": self.tokenizer_name,
            "special_token_id": self.special_token_id,
            "val_num_docs": self.val_num_docs,
            "val_ratio": self.val_ratio,
            "dtype": str(self.dtype),
        }

        # Include filter function bytecode or doc if provided
        if self.filter_fn is not None:
            try:
                components["filter_code"] = inspect.getsource(self.filter_fn)
            except Exception:
                components["filter_code"] = str(self.filter_fn)

        encoded = json.dumps(components, sort_keys=True).encode("utf-8")
        return hashlib.sha256(encoded).hexdigest()[:24]

    def _resolve_dataset_id(self, dataset_arg: Any) -> Optional[str]:
        """Try to resolve dataset identifier string without loading raw data."""
        if isinstance(dataset_arg, str):
            if self._needs_partition(dataset_arg):
                return f"{dataset_arg}:partitioned_{self.split}"
            return f"{dataset_arg}:{self.split}"
        if hasattr(self, "_dataset_id") and isinstance(self._dataset_id, str):
            return self._dataset_id
        if hasattr(self, "_get_dataset_id") and callable(self._get_dataset_id):
            return self._get_dataset_id()
        return None

    def _load_or_build_tokens(self, dataset_arg: Any) -> np.ndarray:
        """Load token stream from cache or tokenize and build cache."""
        os.makedirs(self.cache_dir, exist_ok=True)

        # 1. Fast path: check if cache exists before loading raw data
        dataset_id = self._resolve_dataset_id(dataset_arg)
        cache_file = None
        if dataset_id is not None and self.cache and not self.force_rebuild:
            cache_fp = self._compute_cache_fingerprint(dataset_id)
            cache_file = os.path.join(self.cache_dir, f"{cache_fp}_tokens.npy")
            if os.path.exists(cache_file):
                if self.verbose:
                    print(f"[InMemoryDataset] Instant cache hit! Loading cached token stream: {cache_file}")
                t0 = time.time()
                tokens_np = np.load(cache_file)
                if self.verbose:
                    print(f"[InMemoryDataset] Loaded {len(tokens_np):,} tokens in {time.time() - t0:.4f}s.")
                return tokens_np

        # 2. Cache miss or dynamic dataset: Load raw texts
        texts, actual_dataset_id = self._load_raw_data(dataset_arg)
        dataset_id = dataset_id or actual_dataset_id

        # Check cache again if dataset_id was only resolved after loading raw data
        if self.cache and not self.force_rebuild:
            cache_fp = self._compute_cache_fingerprint(dataset_id)
            cache_file = os.path.join(self.cache_dir, f"{cache_fp}_tokens.npy")
            if os.path.exists(cache_file):
                if self.verbose:
                    print(f"[InMemoryDataset] Loading cached token stream from: {cache_file}")
                t0 = time.time()
                tokens_np = np.load(cache_file)
                if self.verbose:
                    print(f"[InMemoryDataset] Cache loaded in {time.time() - t0:.4f}s.")
                return tokens_np

        # 3. Tokenize dataset in batches
        if self.verbose:
            print(
                f"[InMemoryDataset] Tokenizing {len(texts):,} documents "
                f"(split='{self.split}', tokenizer='{self.tokenizer_name}', threads={self.num_threads})..."
            )

        t0 = time.time()
        tokens_np = self._tokenize_and_assemble(texts)
        dt = time.time() - t0
        tok_per_sec = len(tokens_np) / max(dt, 1e-6)

        if self.verbose:
            print(
                f"[InMemoryDataset] Tokenized {len(tokens_np):,} tokens in {dt:.2f}s "
                f"({tok_per_sec:,.0f} tok/s). Storage: {tokens_np.nbytes / (1024*1024):.2f} MB."
            )

        # 4. Save to cache atomically if caching is enabled
        if self.cache:
            cache_fp = self._compute_cache_fingerprint(dataset_id)
            cache_file = os.path.join(self.cache_dir, f"{cache_fp}_tokens.npy")
            # np.save automatically appends .npy if not present, so ensure tmp filename ends with .npy
            tmp_cache_file = os.path.join(self.cache_dir, f".tmp_{cache_fp}_{os.getpid()}.npy")
            try:
                np.save(tmp_cache_file, tokens_np)
                os.replace(tmp_cache_file, cache_file)
                if self.verbose:
                    print(f"[InMemoryDataset] Saved token stream cache to: {cache_file}")
            except Exception as e:
                warnings.warn(f"Failed to save cache file {cache_file}: {e}")
                if os.path.exists(tmp_cache_file):
                    try:
                        os.remove(tmp_cache_file)
                    except OSError:
                        pass

        return tokens_np

    def _tokenize_and_assemble(self, texts: List[str]) -> np.ndarray:
        """
        Tokenize texts using multi-threaded batch tokenization and assemble
        into a contiguous 1D numpy array with document delimiter tokens.
        """
        batch_size = self.batch_size
        token_batches: List[List[int]] = []
        special_id = self.special_token_id

        total_tokens_count = 0

        for i in tqdm(range(0, len(texts), batch_size), desc="Tokenizing"):
            chunk = texts[i : i + batch_size]
            # Apply transformation and filtering
            filtered_chunk: List[str] = []
            for t in chunk:
                cleaned = self._apply_transform(t)
                if self._apply_filter(cleaned):
                    filtered_chunk.append(cleaned)

            if not filtered_chunk:
                continue

            # Multi-threaded batch tokenization
            encoded_batch = self._tokenize_batch(filtered_chunk)
            for doc_tokens in encoded_batch:
                token_batches.append(doc_tokens)
                total_tokens_count += len(doc_tokens)
                if special_id is not None:
                    total_tokens_count += 1

        if total_tokens_count == 0:
            raise ValueError("All documents were empty or filtered out. 0 tokens found.")

        # Allocate contiguous flat array in memory
        flat_tokens = np.empty(total_tokens_count, dtype=self.dtype)
        ptr = 0

        for doc_tokens in token_batches:
            length = len(doc_tokens)
            flat_tokens[ptr : ptr + length] = doc_tokens
            ptr += length
            if special_id is not None:
                flat_tokens[ptr] = special_id
                ptr += 1

        return flat_tokens

    # -------------------------------------------------------------------------
    # PyTorch Dataset Protocol
    # -------------------------------------------------------------------------
    def __len__(self) -> int:
        """Total number of complete sequence chunks."""
        return self.num_sequences

    def __getitem__(self, index: int):
        """
        Return (x, y) training chunk where y is x shifted by 1 token.
        Both x and y are 1D torch.LongTensor of shape (seq_len,).
        Access is in-memory and zero-copy slicing.
        """
        if index < 0:
            index += self.num_sequences
        if index < 0 or index >= self.num_sequences:
            raise IndexError(f"Index {index} out of range for dataset with {self.num_sequences} sequences.")

        start = index * self.seq_len
        end = start + self.seq_len

        # Fast in-memory slicing
        x = self.tokens[start:end]
        y = self.tokens[start + 1 : end + 1]

        return {
            'input_ids': x.long(),
            'attention_mask': torch.ones_like(x, dtype=torch.long),
            'labels': y.long()
        }

    # -------------------------------------------------------------------------
    # Factory Methods
    # -------------------------------------------------------------------------
    @classmethod
    def splits(
        cls,
        dataset = None,
        tokenizer: Any = "gpt2",
        seq_len: int = 1024,
        **kwargs,
    ) -> Tuple["InMemoryDataset", "InMemoryDataset"]:
        """
        Convenience classmethod to load both train and validation splits.

        Usage:
            train_ds, val_ds = InMemoryDataset.splits("roneneldan/TinyStories", seq_len=1024)
        """
        train_ds = cls(dataset=dataset, tokenizer=tokenizer, seq_len=seq_len, split="train", **kwargs)
        val_ds = cls(dataset=dataset, tokenizer=tokenizer, seq_len=seq_len, split="validation", **kwargs)
        return train_ds, val_ds


# Re-export TinyStoriesDataset for backwards compatibility
from dataset.tinystories import TinyStoriesDataset  # noqa: E402

__all__ = [
    "InMemoryDataset",
    "TinyStoriesDataset",
    "get_special_token_id",
    "get_special_token_str",
    "get_available_special_tokens",
    "get_bos_token_id",
    "get_eos_token_id",
    "get_start_of_text_token_id",
    "get_end_of_text_token_id",
    "get_pad_token_id",
    "get_unk_token_id",
    "get_sep_token_id",
    "get_mask_token_id",
    "get_cls_token_id",
]


