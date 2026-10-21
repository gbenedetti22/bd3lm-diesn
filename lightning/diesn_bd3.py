"""
DiESN (diffusion/diesn.py, unchanged) as the BD3LM backbone, in place of the Transformer of models/dit.py: same
interface as DIT, forward(indices, sigma, sample_mode=False, store_kv=False).

BD3LM trains on the concatenation [x_t, x0] with the attention mask of models/dit.py (block_diff_mask):
  - M_BD (block diagonal): every block reads its whole block, in both directions;
  - M_OBC (offset block causal): the x_t block b reads the x0 blocks < b (its clean context);
  - M_BC (block causal): the x0 block b reads the x0 blocks <= b.
In a Transformer the mask lives in the attention. DIESN moves information along the sequence in three modules (the
rest acts on each position alone): the bidirectional ParalESN and the depthwise convolution (kernel 5) of every
DiESNBlock, and the single attention layer. DiESNBD3 swaps these three modules for versions that apply the same mask
and hold the same parameters and buffers (same state_dict keys as DIESN, as models/fast_esn.use_fast_esn does);
DiESNBlock.forward, DiESNAttention and everything else run as in diesn.py.

Without the mask, the backward reservoir and the attention would read x0 of block b itself (the target) from the
noisy block b. With it, inside a block the denoising stays bidirectional (tests: one block = the original DIESN).

The two halves of [x_t, x0] are computed in two passes, the x0 half first (Algorithm 1 of the paper: the clean pass
that BD3LM adds to training): the x0 pass keeps in every mixer what the x_t blocks read from it (the "x0 context"),
the x_t pass reads it. In sample mode, as DIT, one pass over [clean blocks, block being denoised] with the x0 / x0
part of the mask: the last block then reads exactly what the x_t pass gives block b in training.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from omegaconf import OmegaConf

from diffusion.config import DiESNConfig
from diffusion.diesn import DIESN
from models.dit import block_diff_mask
from models.fast_esn import bidirectional_esn_scan
from models.gpt import apply_rotary_emb
from paralesn.layers import FullyConnFeatureMixer
from paralesn.model import LinearESN

X0, XT = "x0", "xt"  # the two halves of BD3LM's input [x_t, x0]


def _depthwise_conv(windows: torch.Tensor, weight: torch.Tensor, bias: torch.Tensor, block_size: int) -> torch.Tensor:
    """F.conv1d(windows.transpose(1, 2), weight, bias, groups=D) (no padding) on channel-last windows
    (N, block_size + k - 1, D), written as k shifted multiply-adds that torch.compile fuses into one kernel, forward
    and backward (PyTorch's depthwise convolution kernels are slow, the weight gradient above all). (N, block_size, D)."""
    w = weight[:, 0, :]  # (D, k)
    out = bias + windows[:, :block_size] * w[:, 0]
    for j in range(1, w.shape[1]):
        out = out + windows[:, j:j + block_size] * w[:, j]
    return out


_depthwise_conv_compiled = torch.compile(_depthwise_conv, dynamic=True)


class _BlockDiffMixer(nn.Module):
    """A sequence mixer of DIESN under block_diff_mask: which half it is computing (`half`, set by DiESNBD3) and the
    x0 context stored by the last x0 pass (`x0_ctx`, a plain attribute: not in the state_dict)."""

    def __init__(self, block_size: int):
        super().__init__()
        self.block_size = block_size
        self.half = X0
        self.x0_ctx = None


class BlockDiffESN(_BlockDiffMixer):
    """
    The bidirectional ParalESN of a DiESNBlock (paralesn LinearESN: 1 layer, 'cat' merge, fully connected mixer):
        u_t = leaky (x_t W_in + b),  forward reservoir h_t = lambda h_{t-1} + u_t,  backward g_t = lambda g_{t+1} + u_t,
        out_t = tanh(Re[h_t, g_t] W_r^T - Im[h_t, g_t] W_i^T + Re b_mix),
    under block_diff_mask. Inside a block both reservoirs read the whole block (M_BD). The forward reservoir reads
    only earlier positions, so it runs over the whole sequence: it brings the earlier x0 blocks into every block
    (M_BC, M_OBC). The backward reservoir would read later blocks, so it restarts at the end of every block.
    x_t half: block b starts its forward reservoir from the x0 state at the end of block b-1 (the x0 context). By
    linearity, with s_b the first position of block b and k = t - s_b,
        h_t = h_t^block + lambda^(k+1) h^x0_{s_b - 1},
    where h^block restarts from 0 at s_b: one scan of each block on its own and an elementwise correction.
    Same computation as models/fast_esn.FastLinearESN (its Triton scan, TF32 input GEMM, mixer GEMM in the autocast
    dtype) and, as the reference, no gradient through the reservoirs.
    """

    def __init__(self, esn: LinearESN, block_size: int):
        super().__init__(block_size)
        layer = esn.layers[0]
        assert esn.n_layers == 1 and esn.bidirectional and esn.bid_merge == "cat", "unsupported LinearESN"
        assert isinstance(layer.mixer, FullyConnFeatureMixer), "unsupported feature mixer"
        assert layer.reservoir.config.in_initialization in ("gamma", "uniform"), "unsupported input kernel"
        self.layers = esn.layers  # the reference's modules: same buffers, same state_dict keys

    def _scan(self, p: torch.Tensor, n_seq: int) -> torch.Tensor:
        """[Re h | Re g | Im h | Im g] (fp32) of both reservoirs on p = x [Re W_in | Im W_in] cut into n_seq sequences."""
        res = self.layers[0].reservoir
        bias = torch.cat([res.bias.real, res.bias.imag]).float().contiguous()
        lam = torch.view_as_real(res.recurrent_kernel.to(torch.complex64)).contiguous()
        return bidirectional_esn_scan(p.reshape(n_seq, -1, p.shape[-1]), bias, lam, float(res.config.leaky), None,
                                      torch.float32)

    @torch.compiler.disable  # complex states and a Triton custom op: eager also in compiled blocks
    @torch.no_grad()
    def forward(self, x, attention_mask=None, **kwargs):
        assert attention_mask is None, "padding is not supported"
        res, mixer = self.layers[0].reservoir, self.layers[0].mixer
        B, L, D = x.shape
        bs, H = self.block_size, res.n_units
        nb = L // bs
        out_dtype = torch.get_autocast_dtype("cuda") if torch.is_autocast_enabled("cuda") else torch.float32
        with torch.autocast("cuda", enabled=False):  # TF32 input projection, as FastLinearESN
            w_in = torch.cat([res.in_kernel.real, res.in_kernel.imag], dim=1)  # (D, 2H)
            p = (x.float().reshape(B * L, D) @ w_in).view(B, L, 2 * H)
        states = self._scan(p, B * nb).view(B, nb, bs, 4 * H)  # both reservoirs inside every block
        if self.half == X0:
            whole = self._scan(p, B).view(B, nb, bs, 4 * H)  # the forward reservoir over the whole sequence
            states[..., :H] = whole[..., :H]
            states[..., 2 * H:3 * H] = whole[..., 2 * H:3 * H]
            # x0 context: forward state at the end of blocks 0 .. nb-2, which blocks 1 .. nb-1 continue
            self.x0_ctx = torch.complex(whole[:, :-1, -1, :H], whole[:, :-1, -1, 2 * H:3 * H])  # (B, nb-1, H)
        else:
            lam_pow = torch.cumprod(res.recurrent_kernel.to(torch.complex64).expand(bs, H), dim=0)  # lambda^(k+1)
            carry = self.x0_ctx[:, :, None, :] * lam_pow  # (B, nb-1, bs, H)
            states[:, 1:, :, :H] += carry.real
            states[:, 1:, :, 2 * H:3 * H] += carry.imag
        w_mix = torch.cat([mixer.weight.real, -mixer.weight.imag], dim=1)  # (2H, 4H)
        return torch.tanh(F.linear(states.view(B, L, 4 * H).to(out_dtype), w_mix.to(out_dtype),
                                   mixer.bias.real.to(out_dtype)))


class BlockDiffConv(_BlockDiffMixer):
    """
    The depthwise convolution of a DiESNBlock (nn.Conv1d, kernel 5, padding 2: positions t-2 .. t+2) under
    block_diff_mask: inside a block it reads both sides (M_BD); the 2 positions after a block are 0 (later blocks);
    the 2 positions before it come from x0 (earlier blocks): the x0 pass reads its own input, the x_t pass the stored
    x0 positions (the x0 context).
    """

    def __init__(self, conv: nn.Conv1d, block_size: int):
        super().__init__(block_size)
        (k,) = conv.kernel_size
        assert conv.groups == conv.in_channels == conv.out_channels and k % 2 == 1 and conv.padding == ((k - 1) // 2,)
        self.weight, self.bias = conv.weight, conv.bias  # same parameters and state_dict keys
        self.halo = (k - 1) // 2
        self.fused = False  # compile_blocks: _depthwise_conv compiled instead of F.conv1d

    @torch.compiler.disable  # it stores / reads the x0 context: eager also in compiled blocks
    def forward(self, x):  # (B, D, L), as DiESNBlock calls its conv
        B, D, L = x.shape
        bs, r = self.block_size, self.halo
        nb = L // bs
        x = x.transpose(1, 2)  # (B, L, D)
        if self.half == X0:  # positions s_b - r .. s_b - 1 before every block b (0 before the start)
            idx = torch.arange(0, L, bs, device=x.device)[:, None] + torch.arange(r, device=x.device)
            self.x0_ctx = F.pad(x, (0, 0, r, 0))[:, idx]  # (B, nb, r, D)
        windows = torch.cat([self.x0_ctx, x.reshape(B, nb, bs, D), x.new_zeros(B, nb, r, D)], dim=2)
        windows = windows.view(B * nb, bs + 2 * r, D)
        if self.fused:
            return _depthwise_conv_compiled(windows, self.weight, self.bias, bs).view(B, L, D).transpose(1, 2)
        out = F.conv1d(windows.transpose(1, 2), self.weight, self.bias, groups=D)
        return out.view(B, nb, D, bs).permute(0, 2, 1, 3).reshape(B, D, L)


class BlockDiffAttention(_BlockDiffMixer):
    """
    The SelfAttention of diesn.py (attn_impl='softmax': rotary embeddings, QK RMS norm) with block_diff_mask, as the
    DDiTBlock of models/dit.py (SDPA with a boolean mask; FlashAttention 2 has no arbitrary masks):
      x0 pass: the x0 / x0 part of the mask (M_BD + M_BC); stores the x0 keys and values;
      x_t pass: the x_t rows of the mask: its own block (M_BD) and the x0 keys of the blocks before (M_OBC).
    """

    def __init__(self, attn: nn.Module, block_size: int):
        super().__init__(block_size)
        self.n_head, self.head_dim = attn.n_head, attn.head_dim
        self.c_q, self.c_k, self.c_v, self.c_proj = attn.c_q, attn.c_k, attn.c_v, attn.c_proj  # same keys
        self.register_buffer("inv_freq", attn.inv_freq, persistent=False)

    @torch.compiler.disable  # it stores / reads the x0 context: eager also in compiled blocks
    def forward(self, hidden_states, attention_mask=None, **kwargs):
        assert attention_mask is None, "padding is not supported"
        B, T, C = hidden_states.shape
        # as SelfAttention.forward
        q, k, v = (lin(hidden_states).view(B, T, self.n_head, self.head_dim) for lin in (self.c_q, self.c_k, self.c_v))
        freqs = torch.outer(torch.arange(T, device=q.device, dtype=torch.float32), self.inv_freq)
        cos, sin = (f[None, :, None, :].to(q.dtype) for f in (freqs.cos(), freqs.sin()))
        q = F.rms_norm(apply_rotary_emb(q, cos, sin), (self.head_dim,))
        k = F.rms_norm(apply_rotary_emb(k, cos, sin), (self.head_dim,))
        pos = torch.arange(T, device=q.device)
        if self.half == X0:  # queries and keys in the x0 half: indices T .. 2T - 1 of block_diff_mask
            self.x0_ctx = (k, v)
            mask = block_diff_mask(None, None, pos[:, None] + T, pos[None, :] + T, block_size=self.block_size, n=T)
        else:  # x_t queries (0 .. T - 1) on the keys [x_t, x0] (0 .. 2T - 1)
            k, v = torch.cat([k, self.x0_ctx[0]], dim=1), torch.cat([v, self.x0_ctx[1]], dim=1)
            kv_pos = torch.arange(2 * T, device=q.device)
            mask = block_diff_mask(None, None, pos[:, None], kv_pos[None, :], block_size=self.block_size, n=T)
        y = F.scaled_dot_product_attention(q.transpose(1, 2), k.transpose(1, 2), v.transpose(1, 2), attn_mask=mask)
        return self.c_proj(y.transpose(1, 2).reshape(B, T, C)), None, None


class DiESNBD3(nn.Module):
    """
    DIESN with block_diff_mask. `net` has the parameters, buffers and state_dict keys of a DIESN (its own forward is
    not used). config: the BD3LM configuration (model.*, block_size; model.diesn: the other DiESNConfig fields).
    """

    def __init__(self, config, vocab_size: int):
        super().__init__()
        m = config.model
        # hidden_size, n_blocks, cond_dim, dropout, heads from model.*; the other DiESNConfig fields (attn_impl
        # included) from model.diesn, defaults of diffusion/config.py otherwise
        cfg = DiESNConfig(hidden_size=m.hidden_size, n_blocks=m.n_blocks, cond_dim=m.cond_dim, dropout=m.dropout,
                          esn_n_units=m.hidden_size, esn_n_heads=m.n_heads, **OmegaConf.to_container(m.diesn))
        assert cfg.attn_impl == "softmax", (
            "model.diesn.attn_impl must be 'softmax': block_diff_mask needs the softmax attention; the 'gla' variants "
            "of diesn.py are recurrences of fla whose states at the block boundaries are not exposed")
        self.n = m.length
        self.block_size = config.block_size
        self.net = DIESN(OmegaConf.structured(cfg), vocab_size)
        for block in self.net.blocks:
            block.esn = BlockDiffESN(block.esn, self.block_size)
            block.local_conv = BlockDiffConv(block.local_conv, self.block_size)
        self.net.attn.attn = BlockDiffAttention(self.net.attn.attn, self.block_size)
        self._mixers = [mod for mod in self.modules() if isinstance(mod, _BlockDiffMixer)]

    def compile_blocks(self):
        """torch.compile, in place, of the DiESN blocks, the attention layer and the output layer of diesn.py: the
        per-token computation (norms, AdaLN modulation, residuals, dropout, activations, MLP) is fused; the three
        mixers stay eager (the ParalESN scan is an opaque custom op, models/fast_esn.py) and the convolution runs as
        a compiled _depthwise_conv."""
        for mod in self._mixers:
            if isinstance(mod, BlockDiffConv):
                mod.fused = True
        for module in [*self.net.blocks, self.net.attn, self.net.output_layer]:
            module.compile()

    def _trunk(self, inputs, sigma, half):
        """DIESN.forward up to the output layer, with the mixers computing `half`. inputs: token ids (B, T), or the
        soft masks (indices, probs) of TransparencyHead (EmbeddingLayer of diesn.py, path A)."""
        n_tokens = (inputs[0] if isinstance(inputs, tuple) else inputs).shape[1]
        assert n_tokens % self.block_size == 0, "the sequence must be made of whole blocks"
        for mod in self._mixers:
            mod.half = half
        net = self.net
        x = net.vocab_embed(inputs)
        c = F.silu(net.sigma_map(sigma))
        with torch.amp.autocast("cuda", dtype=torch.bfloat16):
            for block in net.blocks:
                x = block(x, c)
            x = net.attn(x, c)
        return x, c

    def _output(self, x, c):
        with torch.amp.autocast("cuda", dtype=torch.bfloat16):
            return self.net.output_layer(x, c)

    def x0_pass(self, x0, sigma):
        """The x0 half: stores the x0 context (its outputs are not used, as in BD3LM)."""
        self._trunk(x0, sigma, X0)

    def xt_pass(self, xt, sigma):
        """The x_t half on the x0 context of the last x0 pass: logits (B, L, V). xt: token ids or soft masks."""
        return self._output(*self._trunk(xt, sigma, XT))

    def forward(self, indices, sigma, sample_mode=False, store_kv=False, last=None):
        """As DIT.forward: training, indices = [x_t, x0] (B, 2L) and the logits of the x_t half (B, L, V); sample
        mode, one sequence of clean blocks followed by the one being denoised (only the logits of the last `last`
        positions if given; inputs may be soft masks)."""
        assert not store_kv, "no KV cache"
        if sample_mode:
            x, c = self._trunk(indices, sigma, X0)
            return self._output(x[:, -last:] if last else x, c)
        self.x0_pass(indices[:, self.n:], sigma)
        return self.xt_pass(indices[:, :self.n], sigma)
