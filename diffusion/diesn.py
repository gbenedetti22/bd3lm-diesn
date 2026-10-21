import math
from typing import Any

import torch
import torch.nn as nn
import torch.nn.functional as F
from fla.layers import GatedLinearAttention
from omegaconf import DictConfig
from torch import Tensor

from models.attention import bidirectional_attention
from models.gpt import apply_rotary_emb
from paralesn.layers import FullyConnMixerConfig
from paralesn.model import LinearESN

from fla.ops.utils import prepare_sequence_ids as build_seq_idx  # token -> sequence index (packed mode)
from torch.utils.checkpoint import checkpoint

# =============================================================================
# Layer Utilities
# =============================================================================

class LayerNorm(nn.Module):
    """Custom LayerNorm that casts to float32 internally for numerical stability."""
    def __init__(self, dim: int):
        super().__init__()
        self.weight = nn.Parameter(torch.ones([dim]))
        self.dim = dim

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        with torch.amp.autocast('cuda', enabled=False):
            x = F.layer_norm(x.float(), [self.dim])
        return x * self.weight[None, None, :]


def modulate(x: torch.Tensor, shift: torch.Tensor,
             scale: torch.Tensor) -> torch.Tensor:
    """
    Adaptive Layer Normalization modulation: x * (1 + scale) + shift.
    """
    return x * (1 + scale.unsqueeze(1)) + shift.unsqueeze(1)


# =============================================================================
# Timestep Embedding
# =============================================================================

class TimestepEmbedder(nn.Module):
    """
    Embeds scalar timestep/sigma values into vector representations.
    Uses sinusoidal encoding followed by a 2-layer MLP.
    """
    def __init__(self, hidden_size: int, frequency_embedding_size: int = 256):
        super().__init__()
        self.mlp = nn.Sequential(
            nn.Linear(frequency_embedding_size, hidden_size, bias=True),
            nn.SiLU(),
            nn.Linear(hidden_size, hidden_size, bias=True))
        self.frequency_embedding_size = frequency_embedding_size

    @staticmethod
    def timestep_embedding(t: torch.Tensor, dim: int,
                           max_period: int = 10000) -> torch.Tensor:
        """Create sinusoidal timestep embeddings (same as in DDPM/DiT)."""
        half = dim // 2
        freqs = torch.exp(
            -math.log(max_period)
            * torch.arange(start=0, end=half, dtype=torch.float32) / half
        ).to(device=t.device)
        args = t[:, None].float() * freqs[None]
        embedding = torch.cat([torch.cos(args), torch.sin(args)], dim=-1)
        if dim % 2:
            embedding = torch.cat(
                [embedding, torch.zeros_like(embedding[:, :1])], dim=-1)
        return embedding

    def forward(self, t: torch.Tensor) -> torch.Tensor:
        t_freq = self.timestep_embedding(t, self.frequency_embedding_size)
        return self.mlp(t_freq)


# =============================================================================
# Embedding Layer (soft-masking)
# =============================================================================

class EmbeddingLayer(nn.Module):
    def __init__(self, dim: int, vocab_dim: int):
        super().__init__()
        self.embedding = nn.Parameter(torch.empty((vocab_dim, dim)))
        nn.init.kaiming_uniform_(self.embedding, a=math.sqrt(5))

    def forward(self, x):
        # Path A: Sparse soft-masking — (indices, probs) tuple
        if isinstance(x, tuple):
            indices, probs = x
            top_embeddings = self.embedding[indices]             # (B, T, k+1, E)
            scaled = top_embeddings * probs.unsqueeze(-1)        # (B, T, k+1, E)
            return torch.sum(scaled, dim=2)                      # (B, T, E)
        # Path B: Standard integer token IDs
        elif x.ndim == 2:
            return self.embedding[x]
        # Path C: Dense probability distribution (slow fallback)
        elif x.ndim == 3:
            return torch.einsum("blv,ve->ble", x.float(),
                                self.embedding.float()).to(x.dtype)
        else:
            raise ValueError("no valid paths")


# =============================================================================
# DiESN Block (Bidirectional ParalESN with AdaLN)
# =============================================================================

def apply_attention_mask(attention_mask: Tensor | None, x: Any) -> Any:
    if attention_mask is not None:
        x = x * attention_mask.unsqueeze(-1)
    return x


class DiESNBlock(nn.Module):
    """
    Bidirectional ParalESN block with AdaLN conditioning.

    Sub-layer 1: Pre-AdaLN -> BiParalESN -> Merge Projection -> Residual
    Sub-layer 2: Pre-AdaLN -> FFN (GELU) -> Residual
    No gating - residual is direct addition.
    """
    def __init__(self, hidden_size: int, cond_dim: int, esn_config: dict, dropout: float, layer_idx=0):
        super().__init__()
        self.norm1 = LayerNorm(hidden_size)
        self.norm2 = LayerNorm(hidden_size)
        self.dropout = dropout

        # AdaLN: 4 modulation vectors (shift1, scale1, gate1, shift2, scale2, gate2)
        self.adaLN_modulation = nn.Linear(cond_dim, 6 * hidden_size, bias=True)
        nn.init.zeros_(self.adaLN_modulation.weight)
        nn.init.zeros_(self.adaLN_modulation.bias)

        # self.local_conv = PackedDepthwiseConv1d(hidden_size, kernel_size=5, padding=2)
        self.local_conv = nn.Conv1d(
            hidden_size, hidden_size,
            kernel_size=5, padding=2,
            groups=hidden_size  # depthwise = leggero
        )

        # Bidirectional ESN (each is an independent frozen reservoir)
        self.esn = LinearESN(**esn_config)

        # Merge projection: input dim depends on bidirectional mode.
        # When bid_merge="cat", the ESN output is 2*n_units;
        # when bid_merge="sum" or not bidirectional, it is n_units.
        # The mixer already reduces to model_dim, so use that.
        esn_out_dim = esn_config.get("model_dim", int(esn_config["n_units"]))
        # self.merge_proj = nn.Linear(esn_out_dim, hidden_size)
        merge_matr = torch.randn(esn_out_dim, hidden_size)
        nn.init.kaiming_uniform_(merge_matr, a=math.sqrt(5))
        self.register_buffer('merge_matr', merge_matr)

        # Trainable FFN
        self.mlp = nn.Sequential(
            nn.Linear(hidden_size, 4 * hidden_size),
            nn.GELU(approximate='tanh'),
            nn.Linear(4 * hidden_size, hidden_size)
        )
        # self.iter = 0
        # self.layer_idx = layer_idx

    def forward(self, x: torch.Tensor, c: torch.Tensor,
                cu_seqlens: torch.Tensor | None = None,
                seq_idx: torch.Tensor | None = None,
                attention_mask: torch.Tensor | None = None) -> torch.Tensor:
        if cu_seqlens is not None:
            # --- Packed-sequence path ---
            # c: (N_seqs, cond_dim), x: (1, TotalTokens, D)
            if seq_idx is None:
                seq_idx = build_seq_idx(cu_seqlens)

            # Compute modulation on N_seqs (cheap), then gather per-token
            mods = self.adaLN_modulation(c)            # (N_seqs, 6*D)
            shift1, scale1, gate1, shift2, scale2, gate2 = mods.chunk(6, dim=-1)

            # Expand each to (1, TotalTokens, D)
            shift1 = shift1[seq_idx].unsqueeze(0)
            scale1 = scale1[seq_idx].unsqueeze(0)
            gate1  = gate1[seq_idx].unsqueeze(0)
            shift2 = shift2[seq_idx].unsqueeze(0)
            scale2 = scale2[seq_idx].unsqueeze(0)
            gate2  = gate2[seq_idx].unsqueeze(0)

            # === ESN Sub-layer ===
            x_norm = self.norm1(x) * (1 + scale1) + shift1
            esn_out = self.esn(x_norm, cu_seqlens=cu_seqlens)       # (1, T, esn_out_dim)
            #conv_out = boundary_patched_conv1d(x_norm.transpose(1, 2), self.local_conv, cu_seqlens).transpose(1, 2)
            conv_out = self.local_conv(x_norm, cu_seqlens=cu_seqlens)
            merged = (esn_out @ self.merge_matr) + F.gelu(conv_out)
            x = x + (gate1 * F.dropout(merged, p=self.dropout, training=self.training))

            # === FFN Sub-layer ===
            x_norm2 = self.norm2(x) * (1 + scale2) + shift2
            mlp_out = self.mlp(x_norm2)
            x = x + (gate2 * F.dropout(mlp_out, p=self.dropout, training=self.training))
        else:
            # --- Standard batched path (unchanged) ---
            # Compute 4 modulation vectors from conditioning c
            shift1, scale1, gate1, shift2, scale2, gate2 = self.adaLN_modulation(c).chunk(6, dim=-1)

            # === ESN Sub-layer ===
            x_norm = modulate(self.norm1(x), shift1, scale1)
            x_norm = apply_attention_mask(attention_mask, x_norm)
                
            esn_out = self.esn(x_norm, attention_mask=attention_mask)  # (B, T, esn_out_dim)
            conv_out = self.local_conv(x_norm.transpose(1, 2)).transpose(1, 2)
            merged = (esn_out @ self.merge_matr) + F.gelu(conv_out)

            merged = apply_attention_mask(attention_mask, merged)
                
            x = x + (gate1.unsqueeze(1) * F.dropout(merged, p=self.dropout, training=self.training))

            # === FFN Sub-layer ===
            x_norm2 = modulate(self.norm2(x), shift2, scale2)

            mlp_out = self.mlp(x_norm2)
            mlp_out = apply_attention_mask(attention_mask, mlp_out)
                
            x = x + (gate2.unsqueeze(1) * F.dropout(mlp_out, p=self.dropout, training=self.training))

        return x


# =============================================================================
# Final Output Layer
# =============================================================================

class DDitFinalLayer(nn.Module):
    """
    Final layer of the DiESN: AdaLN + Linear projection -> vocab logits.
    """
    def __init__(self, hidden_size: int, out_channels: int, cond_dim: int):
        super().__init__()
        self.norm_final = LayerNorm(hidden_size)
        self.linear = nn.Linear(hidden_size, out_channels)
        nn.init.zeros_(self.linear.weight)
        nn.init.zeros_(self.linear.bias)
        self.adaLN_modulation = nn.Linear(cond_dim, 2 * hidden_size, bias=True)
        nn.init.zeros_(self.adaLN_modulation.weight)
        nn.init.zeros_(self.adaLN_modulation.bias)

    def forward(self, x: torch.Tensor, c: torch.Tensor,
                cu_seqlens: torch.Tensor | None = None,
                seq_idx: torch.Tensor | None = None) -> torch.Tensor:
        if cu_seqlens is not None:
            # Packed path: c is (N_seqs, cond_dim)
            if seq_idx is None:
                seq_idx = build_seq_idx(cu_seqlens)
            mods = self.adaLN_modulation(c)  # (N_seqs, 2*D)
            shift, scale = mods.chunk(2, dim=-1)
            shift = shift[seq_idx].unsqueeze(0)  # (1, TotalTokens, D)
            scale = scale[seq_idx].unsqueeze(0)
            x = self.norm_final(x) * (1 + scale) + shift
        else:
            shift, scale = self.adaLN_modulation(c)[:, None].chunk(2, dim=2)
            x = modulate(self.norm_final(x), shift.squeeze(1), scale.squeeze(1))
        return self.linear(x)


class SelfAttention(nn.Module):
    """
    Bidirectional softmax self-attention: every position attends to the whole sequence, on both sides (fla's
    GatedLinearAttention is a causal recurrence: position t reads positions <= t only). Rotary position embeddings
    (relative: right padding does not shift them) and QK norm as in the draft GPT (models/gpt.py), FlashAttention 2
    non-causal with the padding keys excluded (models/attention.py: bidirectional_attention). 4 d^2 parameters (2.36M at
    d = 768; the GatedLinearAttention layer has 2.38M). Same call interface as GatedLinearAttention: (output, None, None).
    """

    def __init__(self, hidden_size: int, num_heads: int, rotary_base: float = 10000.0):
        super().__init__()
        assert hidden_size % num_heads == 0, "hidden_size must be divisible by the number of heads"
        self.n_head, self.head_dim = num_heads, hidden_size // num_heads
        self.c_q = nn.Linear(hidden_size, hidden_size, bias=False)
        self.c_k = nn.Linear(hidden_size, hidden_size, bias=False)
        self.c_v = nn.Linear(hidden_size, hidden_size, bias=False)
        self.c_proj = nn.Linear(hidden_size, hidden_size, bias=False)
        inv_freq = 1.0 / rotary_base ** (torch.arange(0, self.head_dim, 2, dtype=torch.float32) / self.head_dim)
        self.register_buffer("inv_freq", inv_freq, persistent=False)

    def forward(self, hidden_states, attention_mask=None, cu_seqlens=None, **kwargs):
        if cu_seqlens is not None:
            raise NotImplementedError("SelfAttention: packed sequences (cu_seqlens) are not supported")
        B, T, C = hidden_states.shape
        q, k, v = (lin(hidden_states).view(B, T, self.n_head, self.head_dim) for lin in (self.c_q, self.c_k, self.c_v))
        freqs = torch.outer(torch.arange(T, device=q.device, dtype=torch.float32), self.inv_freq)
        cos, sin = (f[None, :, None, :].to(q.dtype) for f in (freqs.cos(), freqs.sin()))
        q = F.rms_norm(apply_rotary_emb(q, cos, sin), (self.head_dim,))
        k = F.rms_norm(apply_rotary_emb(k, cos, sin), (self.head_dim,))
        y = bidirectional_attention(q, k, v, None if attention_mask is None else attention_mask.bool())
        return self.c_proj(y.reshape(B, T, C)), None, None


class BidirectionalGLA(nn.Module):
    """
    Two GatedLinearAttention layers, one reading the sequence left to right and one right to left (the reversed
    sequence), outputs summed: every position reads both sides, in linear time. Each direction has half the heads and
    half the key / value width of the causal layer (the same head dimensions, 32 / 64 at d = 768, 12 heads in all), so
    the two together have its parameter count (2.37M vs 2.38M at d = 768). Right padding becomes left padding in the
    reversed sequence: fla's layer drops the padding positions (attention_mask) in both cases. Same call interface as
    GatedLinearAttention: (output, None, None).
    """

    def __init__(self, hidden_size: int, num_heads: int):
        super().__init__()
        assert num_heads % 2 == 0, "BidirectionalGLA splits the heads between the two directions: an even number"
        kw = dict(num_heads=num_heads // 2, hidden_size=hidden_size, expand_k=0.25, expand_v=0.5)
        self.fwd = GatedLinearAttention(**kw)
        self.bwd = GatedLinearAttention(**kw)

    def forward(self, hidden_states, attention_mask=None, cu_seqlens=None, **kwargs):
        if cu_seqlens is not None:
            raise NotImplementedError("BidirectionalGLA: packed sequences (cu_seqlens) are not supported")
        flip = None if attention_mask is None else attention_mask.flip(1)
        out = self.fwd(hidden_states, attention_mask=attention_mask)[0]
        return out + self.bwd(hidden_states.flip(1), attention_mask=flip)[0].flip(1), None, None


class DiESNAttention(nn.Module):
    def __init__(self, hidden_size: int, esn_n_heads: int, cond_dim: int, attn_impl: str = "gla"):
        """attn_impl: the sequence mixer. 'gla': fla's GatedLinearAttention, causal (position t reads positions <= t
        only; runs owt-v4, owt-v6) | 'gla_bidirectional': BidirectionalGLA, one GLA per direction, linear time |
        'softmax': SelfAttention, bidirectional softmax attention (FlashAttention 2), quadratic time. The same number of
        parameters (2.36-2.38M at d = 768)."""
        super().__init__()
        self.norm = LayerNorm(hidden_size)
        if attn_impl == "gla":
            self.attn = GatedLinearAttention(num_heads=esn_n_heads, hidden_size=hidden_size)
        elif attn_impl == "gla_bidirectional":
            self.attn = BidirectionalGLA(hidden_size, esn_n_heads)
        elif attn_impl == "softmax":
            self.attn = SelfAttention(hidden_size, esn_n_heads)
        else:
            raise ValueError(f"unknown attn_impl '{attn_impl}', expected 'gla', 'gla_bidirectional' or 'softmax'")

        # 3 * hidden_size per shift, scale e gate
        self.adaLN_modulation = nn.Linear(cond_dim, 3 * hidden_size, bias=True)
        nn.init.zeros_(self.adaLN_modulation.weight)
        nn.init.zeros_(self.adaLN_modulation.bias)

    def forward(
            self,
            x: torch.Tensor,
            c: torch.Tensor,
            cu_seqlens: torch.Tensor | None = None,
            seq_idx: torch.Tensor | None = None,
            attention_mask: torch.Tensor | None = None
    ) -> torch.Tensor:
        if cu_seqlens is not None:
            # Packed path: c è (N_seqs, cond_dim)
            if seq_idx is None:
                seq_idx = build_seq_idx(cu_seqlens)
            mods = self.adaLN_modulation(c)  # (N_seqs, 3*D)
            shift, scale, gate = mods.chunk(3, dim=-1)

            shift = shift[seq_idx].unsqueeze(0)  # (1, TotalTokens, D)
            scale = scale[seq_idx].unsqueeze(0)
            gate = gate[seq_idx].unsqueeze(0)

            x_norm = self.norm(x) * (1 + scale) + shift
            return x + gate * self.attn(x_norm, cu_seqlens=cu_seqlens, attention_mask=attention_mask)[0]
        else:
            # Standard batched path: c è (B, cond_dim)
            shift, scale, gate = self.adaLN_modulation(c).chunk(3, dim=-1)
            x_norm = modulate(self.norm(x), shift, scale)
            x_norm = apply_attention_mask(attention_mask, x_norm)
            attn_out = self.attn(x_norm, attention_mask=attention_mask)[0]
            attn_out = apply_attention_mask(attention_mask, attn_out)
            return x + gate.unsqueeze(1) * attn_out

# =============================================================================
# Complete DIESN Backbone
# =============================================================================

class DIESN(nn.Module):
    """
    Parallel Echo State Network (ParalESN) diffusion backbone.
    """
    def __init__(self, config: DictConfig, vocab_size: int):
        super().__init__()
        self.vocab_size = vocab_size

        self.vocab_embed = EmbeddingLayer(config.hidden_size, vocab_size)

        # Timestep conditioning Embedder
        self.sigma_map = TimestepEmbedder(config.cond_dim)

        # Prepare ESN configuration
        from paralesn.layers import LinearReservoirConfig

        # Effective output dim of a single reservoir direction
        esn_effective_units = config.esn_n_units
        if config.esn_bidirectional and config.esn_bid_merge == "cat":
            esn_effective_units = 2 * config.esn_n_units

        esn_config = {
            "in_size": config.hidden_size,
            "n_units": config.esn_n_units,
            "model_dim": esn_effective_units,
            "n_layers": config.esn_n_layers,
            "bidirectional": config.esn_bidirectional,
            "bid_merge": config.esn_bid_merge,
            "reservoir_config": LinearReservoirConfig(
                leaky=config.esn_leaky,
                in_initialization=config.esn_in_initialization,
                in_scaling=config.esn_in_scaling,
                rho_min=config.esn_rho_min,
                rho_max=config.esn_rho_max,
                phase_min=config.esn_phase_min,
                phase_max=config.esn_phase_max,
                n_blocks=config.esn_n_heads,
            ),
            "mixer_config": FullyConnMixerConfig(
                n_units=esn_effective_units,
                model_dim=esn_effective_units,
                mix_scaling=config.esn_mix_scaling,
                mix_bias_scaling=0.0,
            ),
        }

        # Staged blocks
        self.blocks = nn.ModuleList([
            DiESNBlock(config.hidden_size, config.cond_dim, esn_config, config.dropout, layer_idx=i)
            for i, _ in enumerate(range(config.n_blocks))
        ])

        # Output projection
        self.output_layer = DDitFinalLayer(config.hidden_size, vocab_size, config.cond_dim)
        self.attn = DiESNAttention(esn_n_heads=config.esn_n_heads, hidden_size=config.hidden_size, cond_dim=config.cond_dim,
                                   attn_impl=getattr(config, "attn_impl", "gla"))

        # weight tying
        # if config.weight_tying:
        #     self.output_layer.linear.weight = self.vocab_embed.embedding

        self.use_grad_checkpoint = False

    def forward(self, indices, sigma: torch.Tensor, cu_seqlens: torch.Tensor | None = None, attention_mask: torch.Tensor | None = None) -> torch.Tensor:
        x = self.vocab_embed(indices)          # (B, T, D) or (1, TotalTokens, D)
        if attention_mask is not None:
            x = x * attention_mask.unsqueeze(-1)
            
        c = F.silu(self.sigma_map(sigma))      # (B, cond_dim) or (N_seqs, cond_dim)

        # Pre-compute seq_idx once for all blocks (avoids redundant searchsorted)
        seq_idx = build_seq_idx(cu_seqlens) if cu_seqlens is not None else None

        with torch.amp.autocast('cuda', dtype=torch.bfloat16):
            for block in self.blocks:
                if self.use_grad_checkpoint:
                    x = checkpoint(
                        block,
                        x,
                        c,
                        cu_seqlens,
                        seq_idx,
                        attention_mask,
                        use_reentrant=False,
                    )
                else:
                    x = block(x, c, cu_seqlens=cu_seqlens, seq_idx=seq_idx, attention_mask=attention_mask)
                    
            x = self.attn(x, c, cu_seqlens=cu_seqlens, seq_idx=seq_idx, attention_mask=attention_mask)
            x = self.output_layer(x, c, cu_seqlens=cu_seqlens, seq_idx=seq_idx)
        return x

