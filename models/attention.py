"""
Attention backend for the GPT (adapted from nanochat/flash_attention.py) and for the bidirectional attention of the
refiner (`bidirectional_attention`).

nanochat targets Flash Attention 3 (Hopper). On this project's hardware (A100, sm80)
FA3 is not available, but Flash Attention 2 is installed and exposes the same two
entry points with the same (B, T, H, D) layout:

    flash_attn_func(q, k, v, causal=True, window_size=(left, right))
    flash_attn_with_kvcache(q, k_cache, v_cache, k=k, v=v, cache_seqlens=..., causal=True, window_size=...)

FA2 is used when running on CUDA with bf16/fp16 activations, otherwise we fall back to
PyTorch SDPA (CPU tests, fp32). The fallback is a direct port of nanochat's SDPA path.
"""

import os

import torch
import torch.nn.functional as F

try:  # Flash Attention 2
    from flash_attn import flash_attn_func as _fa2_func
    from flash_attn import flash_attn_varlen_func as _fa2_varlen
    from flash_attn import flash_attn_with_kvcache as _fa2_kvcache
    HAS_FA2 = True
except Exception:  # pragma: no cover - depends on the environment
    HAS_FA2 = False

# Set DRAFT_REFINE_ATTN=sdpa to force the fallback (useful for debugging / equivalence tests).
_FORCE_SDPA = os.environ.get("DRAFT_REFINE_ATTN", "").lower() == "sdpa"


def _use_fa2(q: torch.Tensor) -> bool:
    return (
        HAS_FA2
        and not _FORCE_SDPA
        and q.is_cuda
        and q.dtype in (torch.bfloat16, torch.float16)
        and q.size(-1) % 8 == 0
        and q.size(-1) <= 256
    )


def _sdpa_attention(q, k, v, window_size, enable_gqa):
    """SDPA attention with sliding window support. q, k, v are (B, H, T, D)."""
    Tq = q.size(2)
    Tk = k.size(2)
    window = window_size[0]

    # Full context, same length
    if (window < 0 or window >= Tq) and Tq == Tk:
        return F.scaled_dot_product_attention(q, k, v, is_causal=True, enable_gqa=enable_gqa)

    # Single token generation
    if Tq == 1:
        if 0 <= window < Tk:
            start = max(0, Tk - (window + 1))
            k = k[:, :, start:, :]
            v = v[:, :, start:, :]
        return F.scaled_dot_product_attention(q, k, v, is_causal=False, enable_gqa=enable_gqa)

    # Chunked inference (Tq != Tk) or sliding window: explicit boolean mask
    device = q.device
    row_idx = (Tk - Tq) + torch.arange(Tq, device=device).unsqueeze(1)
    col_idx = torch.arange(Tk, device=device).unsqueeze(0)
    mask = col_idx <= row_idx
    if 0 <= window < Tk:
        mask = mask & ((row_idx - col_idx) <= window)
    return F.scaled_dot_product_attention(q, k, v, attn_mask=mask, enable_gqa=enable_gqa)


def flash_attn_func(q, k, v, causal=True, window_size=(-1, -1)):
    """Training attention (no KV cache). q, k, v: (B, T, H, D). Returns (B, T, H, D)."""
    if _use_fa2(q):
        return _fa2_func(q, k, v, causal=causal, window_size=window_size)
    q, k, v = q.transpose(1, 2), k.transpose(1, 2), v.transpose(1, 2)
    y = _sdpa_attention(q, k, v, window_size, enable_gqa=q.size(1) != k.size(1))
    return y.transpose(1, 2)


def bidirectional_attention(q, k, v, key_mask=None):
    """
    Full (non-causal) attention: every position attends to the whole sequence. q, k, v: (B, T, H, D); key_mask: (B, T)
    bool, False on padding (None = no padding), whose keys are excluded (FA2's varlen kernel on the valid tokens only;
    the outputs at padding positions are 0). Returns (B, T, H, D).
    """
    if key_mask is not None and bool(key_mask.all()):
        key_mask = None
    if _use_fa2(q):
        if key_mask is None:
            return _fa2_func(q, k, v, causal=False)
        B, T, H, D = q.shape
        idx = key_mask.flatten().nonzero().squeeze(1)
        lens = key_mask.sum(1, dtype=torch.int32)
        cu = F.pad(torch.cumsum(lens, 0, dtype=torch.int32), (1, 0))
        n = int(lens.max())
        out = _fa2_varlen(*(t.reshape(B * T, H, D)[idx] for t in (q, k, v)), cu, cu, n, n, causal=False)
        return q.new_zeros(B * T, H, D).index_copy_(0, idx, out).view(B, T, H, D)
    mask = None if key_mask is None else key_mask.bool()[:, None, None, :]
    y = F.scaled_dot_product_attention(q.transpose(1, 2), k.transpose(1, 2), v.transpose(1, 2), attn_mask=mask)
    y = y.transpose(1, 2)
    return y if key_mask is None else y * key_mask[:, :, None, None].to(y.dtype)


def flash_attn_with_kvcache(q, k_cache, v_cache, k=None, v=None, cache_seqlens=None,
                            causal=True, window_size=(-1, -1)):
    """
    Inference attention with an in-place updated KV cache (FA2/FA3 semantics).

    q: (B, T_new, H, D); k_cache/v_cache: (B, T_max, H_kv, D); k/v: (B, T_new, H_kv, D);
    cache_seqlens: (B,) int32 current positions (all rows share the same position here).
    """
    if _use_fa2(q):
        return _fa2_kvcache(q, k_cache, v_cache, k=k, v=v, cache_seqlens=cache_seqlens,
                            causal=causal, window_size=window_size)
    B, T_new, H, D = q.shape
    pos = int(cache_seqlens[0].item())
    if k is not None and v is not None:
        k_cache[:, pos:pos + T_new] = k
        v_cache[:, pos:pos + T_new] = v
    end_pos = pos + T_new
    q_s = q.transpose(1, 2)
    k_s = k_cache[:, :end_pos].transpose(1, 2)
    v_s = v_cache[:, :end_pos].transpose(1, 2)
    y = _sdpa_attention(q_s, k_s, v_s, window_size, enable_gqa=q_s.size(1) != k_s.size(1))
    return y.transpose(1, 2)
