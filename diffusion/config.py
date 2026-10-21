"""Configuration of the DIESN backbone (the fields diffusion/diesn.py reads), with the defaults of the original project."""

from dataclasses import dataclass


@dataclass
class DiESNConfig:
    hidden_size: int = 384
    n_blocks: int = 6
    cond_dim: int = 128
    dropout: float = 0.0
    esn_n_units: int = 384  # reservoir units per direction
    esn_n_layers: int = 1
    esn_bidirectional: bool = True
    esn_bid_merge: str = "cat"
    esn_n_heads: int = 6
    esn_leaky: float = 0.9
    esn_in_initialization: str = "gamma"
    esn_in_scaling: float = 0.36681373003472423
    esn_rho_min: float = 0.4  # reservoir spectral radii: how far the context reaches
    esn_rho_max: float = 0.9
    esn_phase_min: float = 0.0
    esn_phase_max: float = 0.19634954084936207
    esn_mix_scaling: float = 0.1
    attn_impl: str = "gla"  # final attention layer: 'gla' (causal) | 'gla_bidirectional' | 'softmax' (bidirectional)
