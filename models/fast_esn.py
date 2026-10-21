"""
Faster implementation of the bidirectional ParalESN layer of the DiESN blocks: paralesn.model.LinearESN with one
layer, bidirectional 'cat' merge and the fully connected feature mixer (the configuration of diesn.py).

paralesn/ (provided code) is unchanged and stays the reference: `FastLinearESN` wraps a LinearESN, shares its
buffers (same random reservoir, same state_dict keys) and computes the same function, without gradient as the
reference (the gradient does not cross the reservoirs):
    u_t   = leaky (x_t W_in + b)                          W_in (D, H), b (H,) complex, x real
    h_t   = f_t h_{t-1} + u_t   (forward)                 f_t = lambda (diagonal recurrent kernel); at padding
    g_t   = f_t g_{t+1} + u_t   (backward)                f_t = 0 and u_t = 0, as the reference does
    out_t = tanh(Re[h_t, g_t] W_r^T - Im[h_t, g_t] W_i^T + Re b_mix)      W = W_r + i W_i (2H, 2H)
How it differs from the reference computation:
  - x is real, so x W_in is one real GEMM x [Re W_in | Im W_in] in TF32, like the complex GEMM of the reference,
    instead of two complex GEMMs (one per direction, on an input with zero imaginary part): 4x fewer FLOPs;
  - both directions run in one Triton kernel that reads u from the GEMM output, keeps the constant gate in
    registers (the reference materializes it as a (B, H, T) complex tensor per direction for the accelerated-scan
    kernel) and writes the mixer input [Re h | Re g | Im h | Im g] in the dtype of the mixer GEMM (bf16 under the
    autocast of DiESN, as the reference's mixer casts it): no complex casts, flips, transposes or concatenations;
  - the mixer is one GEMM with the weight [W_r | -W_i] instead of two GEMMs and a subtraction.
Results agree with the reference to the rounding of the bf16 mixer (tests/test_fast_esn.py).
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import triton
import triton.language as tl

from paralesn.layers._mixer import FullyConnFeatureMixer
from paralesn.model import LinearESN


@triton.jit
def _complex_mul(ar, ai, br, bi):
    return ar * br - ai * bi, ar * bi + ai * br


@triton.jit
def _combine(xl_r, xl_i, fl_r, fl_i, xr_r, xr_i, fr_r, fr_i):
    """(x, f) of a prefix followed by the element (xr, fr): x = x_l f_r + x_r, f = f_l f_r."""
    x_r, x_i = _complex_mul(xl_r, xl_i, fr_r, fr_i)
    f_r, f_i = _complex_mul(fl_r, fl_i, fr_r, fr_i)
    return x_r + xr_r, x_i + xr_i, f_r, f_i


@triton.jit
def _bidirectional_scan_kernel(
    P, BIAS, LAM, MASK, OUT, T, H, leaky,
    stride_pb, stride_pt, stride_ob, stride_ot, stride_mb,
    HAS_MASK: tl.constexpr, BLOCK_T: tl.constexpr, BLOCK_C: tl.constexpr,
):
    """
    One program = one sequence, BLOCK_C channels, one direction (program_id 2: 0 forward, 1 backward), all time
    steps in chunks of BLOCK_T (an associative scan per chunk plus the carry of the previous chunk).
    P (B, T, 2H) fp32: [Re | Im] of x W_in; BIAS (2H,): [Re b | Im b]; LAM (H, 2): lambda as (re, im);
    MASK (B, T) int8 (HAS_MASK); OUT (B, T, 4H): [Re h | Re g | Im h | Im g].
    The backward direction walks the time steps from the end: its chunks are read in reversed time order, so the
    same forward scan along the rows solves g_t = f_t g_{t+1} + u_t.
    """
    b = tl.program_id(0)
    c = tl.program_id(1) * BLOCK_C + tl.arange(0, BLOCK_C)
    direction = tl.program_id(2)
    cmask = c < H
    lam_r = tl.load(LAM + 2 * c, mask=cmask, other=0.0)
    lam_i = tl.load(LAM + 2 * c + 1, mask=cmask, other=0.0)
    bias_r = tl.load(BIAS + c, mask=cmask, other=0.0)
    bias_i = tl.load(BIAS + H + c, mask=cmask, other=0.0)
    carry_r = tl.zeros((BLOCK_C,), dtype=tl.float32)
    carry_i = tl.zeros((BLOCK_C,), dtype=tl.float32)
    rows = tl.arange(0, BLOCK_T)
    for k in range(tl.cdiv(T, BLOCK_T)):
        s = k * BLOCK_T + rows  # position in the processing order
        t = tl.where(direction == 0, s, T - 1 - s)  # time step
        smask = s < T
        m2 = smask[:, None] & cmask[None, :]
        p = P + b * stride_pb + t[:, None] * stride_pt + c[None, :]
        u_r = leaky * (tl.load(p, mask=m2, other=0.0) + bias_r[None, :])
        u_i = leaky * (tl.load(p + H, mask=m2, other=0.0) + bias_i[None, :])
        f_r = tl.broadcast_to(lam_r[None, :], (BLOCK_T, BLOCK_C))
        f_i = tl.broadcast_to(lam_i[None, :], (BLOCK_T, BLOCK_C))
        if HAS_MASK:  # padding: zero input and zero gate (the state restarts from 0 after it)
            valid = (tl.load(MASK + b * stride_mb + t, mask=smask, other=0) != 0)[:, None]
            u_r = tl.where(valid, u_r, 0.0)
            u_i = tl.where(valid, u_i, 0.0)
            f_r = tl.where(valid, f_r, 0.0)
            f_i = tl.where(valid, f_i, 0.0)
        u_r = tl.where(m2, u_r, 0.0)  # rows past the end of the sequence (last chunk): no input
        u_i = tl.where(m2, u_i, 0.0)
        x_r, x_i, cf_r, cf_i = tl.associative_scan((u_r, u_i, f_r, f_i), axis=0, combine_fn=_combine)
        add_r, add_i = _complex_mul(carry_r[None, :], carry_i[None, :], cf_r, cf_i)
        x_r += add_r
        x_i += add_i
        o = OUT + b * stride_ob + t[:, None] * stride_ot + direction * H + c[None, :]
        tl.store(o, x_r.to(OUT.dtype.element_ty), mask=m2)
        tl.store(o + 2 * H, x_i.to(OUT.dtype.element_ty), mask=m2)
        last = (rows == BLOCK_T - 1)[:, None]
        carry_r = tl.sum(tl.where(last, x_r, 0.0), axis=0)
        carry_i = tl.sum(tl.where(last, x_i, 0.0), axis=0)


def _block_c(B, H, device):
    """Channels per program: the widest tile that still gives every SM ~4 programs (narrow tiles for small batches,
    e.g. batch-1 inference on long sequences). BLOCK_T 16, 2 warps: the fastest tiles measured on A100."""
    n_sm = torch.cuda.get_device_properties(device).multi_processor_count
    for block_c in (64, 32):
        if B * triton.cdiv(H, block_c) * 2 >= 4 * n_sm:
            return block_c
    return 16


def _launch_scan(p, bias, lam, leaky, mask, out_dtype, block_t=16, block_c=None, num_warps=2):
    B, T, H2 = p.shape
    H = H2 // 2
    block_c = block_c or _block_c(B, H, p.device)
    out = torch.empty(B, T, 2 * H2, device=p.device, dtype=out_dtype)
    if mask is not None:
        mask = mask.to(torch.int8).contiguous()
    grid = (B, triton.cdiv(H, block_c), 2)
    _bidirectional_scan_kernel[grid](
        p, bias, lam, mask if mask is not None else p, out, T, H, float(leaky),
        p.stride(0), p.stride(1), out.stride(0), out.stride(1), mask.stride(0) if mask is not None else 0,
        HAS_MASK=mask is not None, BLOCK_T=block_t, BLOCK_C=block_c, num_warps=num_warps,
    )
    return out


# an opaque operator for torch.compile (the DiESN blocks are compiled): Dynamo does not trace into the kernel, and
# knowing that it mutates nothing it does not copy its inputs
@torch.library.custom_op("diesn::bidirectional_esn_scan", mutates_args=())
def bidirectional_esn_scan(p: torch.Tensor, bias: torch.Tensor, lam: torch.Tensor, leaky: float,
                           mask: torch.Tensor | None, out_dtype: torch.dtype) -> torch.Tensor:
    """[Re h | Re g | Im h | Im g] (B, T, 4H) of the two reservoir directions from p = x [Re W_in | Im W_in]."""
    return _launch_scan(p, bias, lam, leaky, mask, out_dtype)


@bidirectional_esn_scan.register_fake
def _(p, bias, lam, leaky, mask, out_dtype):
    B, T, H2 = p.shape
    return p.new_empty((B, T, 2 * H2), dtype=out_dtype)


class FastLinearESN(nn.Module):
    """
    Drop-in replacement of a paralesn LinearESN (n_layers=1, bidirectional, bid_merge='cat', FullyConnFeatureMixer,
    'gamma' / 'uniform' input kernel) with the same buffers and state_dict keys. Other configurations, packed
    sequences (cu_seqlens), last_state_only and CPU tensors are delegated to the reference module.
    """

    def __init__(self, esn: LinearESN):
        super().__init__()
        layer = esn.layers[0]
        assert esn.n_layers == 1 and esn.bidirectional and esn.bid_merge == "cat", "unsupported LinearESN"
        assert isinstance(layer.mixer, FullyConnFeatureMixer), "unsupported feature mixer"
        assert layer.reservoir.config.in_initialization in ("gamma", "uniform"), "unsupported input kernel"
        self.layers = esn.layers  # the reference's modules: same buffers, same state_dict keys
        self.__dict__["reference"] = esn  # not a registered submodule (it shares self.layers)

    @torch.no_grad()
    def forward(self, x, last_state_only=False, cu_seqlens=None, attention_mask=None):
        if last_state_only or cu_seqlens is not None or not x.is_cuda:
            return eager_reference(self.reference)(x, last_state_only=last_state_only, cu_seqlens=cu_seqlens,
                                                   attention_mask=attention_mask)
        reservoir, mixer = self.layers[0].reservoir, self.layers[0].mixer
        B, T, D = x.shape
        # the dtype in which the reference's mixer (autocast F.linear) runs
        out_dtype = torch.get_autocast_dtype("cuda") if torch.is_autocast_enabled("cuda") else torch.float32
        with torch.autocast("cuda", enabled=False):  # TF32 like the reference's complex GEMM, not bf16
            w_in = torch.cat([reservoir.in_kernel.real, reservoir.in_kernel.imag], dim=1)  # (D, 2H)
            p = (x.float().reshape(B * T, D) @ w_in).view(B, T, -1)
        bias = torch.cat([reservoir.bias.real, reservoir.bias.imag]).float().contiguous()
        lam = torch.view_as_real(reservoir.recurrent_kernel.to(torch.complex64)).contiguous()
        states = bidirectional_esn_scan(p, bias, lam, float(reservoir.config.leaky), attention_mask, out_dtype)
        w_mix = torch.cat([mixer.weight.real, -mixer.weight.imag], dim=1)  # (2H, 4H)
        out = torch.tanh(F.linear(states, w_mix.to(states.dtype), mixer.bias.real.to(states.dtype)))
        if attention_mask is not None:
            out = out * attention_mask.unsqueeze(-1)
        return out


def eager_reference(esn: nn.Module):
    """The forward of a reference LinearESN excluded from torch.compile. Dynamo cannot analyze the accelerated-scan
    Triton kernel ("assuming every input is mutated") and the compiled reservoir returns wrong states (max |diff|
    0.25 on outputs of at most 0.2); run eagerly inside a compiled block it is exact."""
    return torch.compiler.disable(esn.forward)


def use_fast_esn(backbone: nn.Module) -> int:
    """Replace the LinearESN of every DiESN block of `backbone` with a FastLinearESN; returns how many."""
    n = 0
    for block in backbone.blocks:
        if isinstance(block.esn, LinearESN):
            block.esn = FastLinearESN(block.esn)
            n += 1
    return n
