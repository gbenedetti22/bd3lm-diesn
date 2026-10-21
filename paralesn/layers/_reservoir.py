"""(linear) Reservoir for the LinearESN architecture."""

__all__ = ["LinearReservoirConfig", "LinearReservoir"]

from dataclasses import dataclass

import numpy as np
import torch
import torch.nn as nn
from torch import Tensor

from paralesn.utils import (
    # associative_scan,
    init_bias,
    init_diagonal_kernel,
    init_multihead_diagonal_kernel,
    init_kernel,
)

from accelerated_scan.complex import scan

@dataclass
class LinearReservoirConfig:
    """Hyperparameters' configuration for the linear reservoir layer.

    Attributes
    ----------
    leaky : float
        Leaky rate (0, 1].
    in_initialization : str
        Initialization method for the input kernel. Options are 'gamma', 'uniform', or 
        'ring'.
    in_scaling : float
        Weight scaling for the input kernel.
    bias_scaling : float
        Bias scaling for the recurrent kernel bias.
    rho_min : float
        Minimum spectral radius for sampling diagonal values of the recurrent kernel.
    rho_max : float
        Maximum spectral radius for sampling diagonal values of the recurrent kernel.
    phase_min : float
        Minimum phases for sampling diagonal values of the recurrent kernel.
    phase_max : float
        Maximum phases for sampling diagonal values of the recurrent kernel.
    n_blocks : int | None
        Number of "multi-head" blocks for partitioning the diagonal kernel.
        Each block is initialized with a distinct sub-range of spectral radii
        and phases (linearly interpolated from the global ranges), mimicking the
        diversity of Multi-Head Attention.  When None, the standard uniform
        initialization is used (no partitioning).
    """

    leaky: float = 1.0
    in_initialization: bool = "gamma"
    in_scaling: float | None = None
    bias_scaling: float = 1.0

    # for the complex-valued recurrent kernel
    rho_min: float = 0.0
    rho_max: float = 1.0
    phase_min: float = 0.0
    phase_max: float = 2 * np.pi

    # multi-head diagonal initialization
    n_blocks: int | None = None


class LinearReservoir(nn.Module):
    """Implementation of a LinearReservoir layer of LinearESN.

    Parameters
    ----------
    in_size : int
        Number of expected input features.
    n_units : int
        Number of hidden units in the reservoir.
    config : LinearReservoirConfig
        Configuration for the linear reservoir layer.

    Attributes
    ----------
    recurrent_kernel : Tensor
        The recurrent weight matrix of the reservoir.
    in_kernel : Tensor
        The input weight matrix of the reservoir.
    bias : Tensor
        The bias vector of the reservoir.
    """

    def __init__(
        self,
        in_size: int,
        n_units: int,
        config: LinearReservoirConfig,
    ) -> None:
        """
        Parameters
        ----------
        in_size : int
            Number of expected input features.
        n_units : int
            Number of hidden units in the reservoir.
        config : LinearReservoirConfig
            Configuration for the linear reservoir layer.
        """
        super().__init__()
        self.n_units = n_units
        self.config = config

        # --- Recurrent kernel: standard or multi-head ---
        if config.n_blocks is not None:
            recurrent_kernel = init_multihead_diagonal_kernel(
                M=n_units,
                n_blocks=config.n_blocks,
                leaky=config.leaky,
                rho_min=config.rho_min,
                rho_max=config.rho_max,
                phase_min=config.phase_min,
                phase_max=config.phase_max,
            )
        else:
            recurrent_kernel = init_diagonal_kernel(
                M=n_units,
                leaky=config.leaky,
                rho_min=config.rho_min,
                rho_max=config.rho_max,
                phase_min=config.phase_min,
                phase_max=config.phase_max,
            )
        self.register_buffer("recurrent_kernel", recurrent_kernel)

        if config.in_initialization in ["gamma", "uniform"]:
            in_kernel = init_kernel(
                M=in_size, 
                N=n_units, 
                initialization=config.in_initialization,
                scaling=config.in_scaling, 
                lambdas=recurrent_kernel
            )
            self.register_buffer("in_kernel", in_kernel)
        elif config.in_initialization == "ring":
            self.in_kernel = lambda x: self.config.in_scaling * torch.roll(
                x, shifts=1, dims=-1
            )
        else:
            raise ValueError(
                f"Unknown input kernel initialization: {config.in_initialization}"
            )

        bias = init_bias(M=n_units, scaling=config.bias_scaling)
        self.register_buffer("bias", bias)

    def _init_hidden_state(self, batch_size: int, device: torch.device) -> Tensor:
        """Initialize a zero-valued hidden state for the reservoir.

        Parameters
        ----------
        batch_size : int
            Size of the batch for which to initialize the hidden state.
        device : torch.device
            Device to create the tensor on.

        Returns
        -------
        Tensor
            A tensor of shape (batch_size (B), n_units (H)).
        """
        return torch.zeros(batch_size, self.n_units, dtype=torch.complex64, device=device)
    
    def _compute_last_state(self, x: torch.Tensor, h_0: torch.Tensor) -> torch.Tensor:
        """Compute (only) the last hidden state directly by unrolling the linear recurrence:

            h(t) = A^t * h(0) + Σ(i=1 to t) A^(t-i) * [B * x(i) + bias]
        
        Args:
            x: Input sequence of shape (batch_size, seq_len, input_dim)
            h_0: Initial hidden state of shape (batch_size, hidden_dim)
        
        Returns:
            Final hidden state of shape (batch_size, hidden_dim)
        """
        seqlen = x.shape[1] # timesteps t

        # compute input projections W_in @ x + bias / scaling * torch.roll(x) + bias
        # shape should be (B, T, H)
        if self.config.in_initialization == "ring":
            Bu_elements = self.config.leaky * (self.in_kernel(x) + self.bias)
        else:
            print(f"Dtype di x: {x.dtype}, Dtype di in_kernel: {self.in_kernel.dtype}")
            Bu_elements = self.config.leaky * (x @ self.in_kernel + self.bias)
        
        # First term: A^t * h(0)
        A_pow_t = torch.pow(self.recurrent_kernel, seqlen)
        Ah_0 = A_pow_t.unsqueeze(0) * h_0
        
        # Summation terms: Σ(i=1 to t) A^(t-i) * B * x(i)        
        # Create powers of A from 0 to t-1
        powers = torch.arange(seqlen, device=self.recurrent_kernel.device).unsqueeze(1)
        A_pows = torch.pow(self.recurrent_kernel.unsqueeze(0), powers)  # (T, H)
        
        # Reverse for alignment
        Bx_reversed = torch.flip(Bu_elements, dims=[1])  # (B, T, H)
        summation_term = torch.sum(A_pows.unsqueeze(0).real * Bx_reversed, dim=1)
        
        last_state = Ah_0 + summation_term
        return last_state

    # @torch.compiler.disable
    # def _forward_scan(
    #     self,
    #     x: Tensor,
    #     h_prev: Tensor | None = None,
    # ) -> Tensor:
    #     """Forward pass via parallel associative scan (fast!).
    #
    #     Only handles inputs of shape (batch_size (B), seq_len (T), n_features (N)).
    #
    #     Code adapted from
    #     - https://github.com/DecodEPFL/SSM/blob/main/lru/lru.py
    #
    #     which in turns borrows heavily from
    #     - https://colab.research.google.com/drive/1RgIv_3WA
    #     - https://github.com/i404788/s5-pytorch
    #
    #     Parameters
    #     ----------
    #     x : Tensor
    #         Input tensor of shape (B, T, Nx).
    #     h_prev : Optional[Tensor], optional
    #         Hidden state at the previous time step, of shape (B, hidden_size (H)). If
    #         None, it is initialized to a zero vector.
    #     """
    #     lambdas = self.recurrent_kernel  # diag elements of the recurrent kernel
    #
    #     # repeat lambdas to shape (T, H)
    #     lambda_elements = lambdas.tile(x.shape[1], 1)
    #
    #     # compute input projections W_in @ x + bias / scaling * torch.roll(x) + bias
    #     # shape should be (B, T, H)
    #     if self.config.in_initialization == "ring":
    #         Bu_elements = self.config.leaky * (self.in_kernel(x) + self.bias)
    #     else:
    #         Bu_elements = self.config.leaky * (x @ self.in_kernel + self.bias)
    #
    #     if h_prev is not None:
    #         Bu_elements[:, 0, :] = Bu_elements[:, 0, :] + (lambdas * h_prev)
    #     # Vmap the associative scan since Bu_elements is a batch of B sequences.
    #     # recall that lambda_elements has been repeated L times to (T, H),
    #     # while Bu_seq has shape (B, T, H)
    #     inner_state_fn = lambda Bu_seq: associative_scan(
    #         elems=(lambda_elements, Bu_seq),
    #     )[1]
    #
    #     # inner_states will be of shape (B, T, H)
    #     inner_states = torch.vmap(inner_state_fn)(Bu_elements)
    #     return inner_states

    def _forward_scan(
            self,
            x: Tensor,
            h_prev: Tensor | None = None,
            cu_seqlens: Tensor | None = None,
            attention_mask: Tensor | None = None,
    ) -> Tensor:
        #     Only handles inputs of shape (batch_size (B), seq_len (T), n_features (N)).

        B, T, _ = x.shape
        lambdas = self.recurrent_kernel

        if self.config.in_initialization == "ring":
            Bu_elements = self.config.leaky * (self.in_kernel(x) + self.bias)
        else:
            Bu_elements = self.config.leaky * (x @ self.in_kernel + self.bias)

        if h_prev is not None:
            Bu_elements[:, 0, :] = Bu_elements[:, 0, :] + (lambdas * h_prev)

        if attention_mask is not None:
            Bu_elements = Bu_elements * attention_mask.unsqueeze(-1)

        # Build gates in (B, H, T) layout: expand lambdas (H,) -> (1, H, 1) -> (B, H, T)
        gates = lambdas.unsqueeze(1).expand(-1, T).unsqueeze(0).expand(B, -1, -1).contiguous()  # (B, H, T)

        if attention_mask is not None:
            mask_bht = attention_mask.unsqueeze(1).expand(-1, gates.shape[1], -1)
            gates = torch.where(mask_bht, gates, torch.zeros_like(gates))

        # Zero gates at sequence boundaries to reset recurrent state.
        # Setting a_t = 0 yields x_t = b_t (no dependence on prior state).
        if cu_seqlens is not None:
            boundary_positions = cu_seqlens[:-1].long()
            gates[:, :, boundary_positions] = 1e-6

        tokens = Bu_elements.transpose(1, 2).contiguous()  # (B, H, T)

        inner_states = scan(gates, tokens)  # (B, H, T)
        inner_states = inner_states.transpose(1, 2)  # (B, T, H) — view, no copy
        return inner_states

    def _backward_scan(self, x: Tensor, cu_seqlens: Tensor | None = None, attention_mask: Tensor | None = None) -> Tensor:
        """Backward (right-to-left) pass via the same parallel scan.

        Flips the input temporally, runs the forward scan with a fresh
        zero hidden state, then flips the output back so that positions
        are aligned with the forward pass.

        When *cu_seqlens* is provided, flipping is done **within** each
        sub-sequence so that tokens from different sequences are never
        mixed.  Boundary gates are zeroed in the forward scan to prevent
        cross-sequence information leakage.

        Parameters
        ----------
        x : Tensor
            Input tensor of shape (B, T, N).
        cu_seqlens : Tensor, optional
            Cumulative sequence lengths for packed mode.

        Returns
        -------
        Tensor
            Hidden states of shape (B, T, H), temporally aligned with the
            original (un-flipped) input.
        """
        x_flip = torch.flip(x, dims=[1])

        if cu_seqlens is not None:
            cu_seqlens = cu_seqlens[-1] - cu_seqlens.flip(0)  # boundaries of the time-reversed sequences

        if attention_mask is not None:
            attention_mask = torch.flip(attention_mask, dims=[1])

        h0 = self._init_hidden_state(x.shape[0], x.device)
        bwd_states = self._forward_scan(x_flip, h_prev=h0, cu_seqlens=cu_seqlens, attention_mask=attention_mask)

        return torch.flip(bwd_states, dims=[1])

    @torch.no_grad()
    def forward(
        self,
        x: Tensor,
        h_prev: Tensor | None = None,
        last_state_only: bool = False,
        bidirectional: bool = False,
        bid_merge: str = "cat",
        cu_seqlens: Tensor | None = None,
        attention_mask: Tensor | None = None,
    ) -> Tensor:
        """Forward pass of the linear reservoir layer.

        Parameters
        ----------
        x : Tensor
            Input tensor of shape (batch_size (B), seq_len (T), n_features (N)).
            The input tensor is expected to be complex-valued.
        h_prev : Optional[Tensor], optional
            Hidden state at the previous time step, of shape (B, n_units (H)). If
            None, it is initialized to a zero vector.
        last_state_only : bool, optional
            If True, only the last hidden state is returned (forward-only).
        bidirectional : bool, optional
            If True, run both a forward and backward scan and merge the
            results.
        bid_merge : str, optional
            How to merge forward and backward states. ``"cat"`` concatenates
            along the hidden dimension (output is 2*H); ``"sum"`` adds them
            element-wise (output is H).
        cu_seqlens : Tensor, optional
            Cumulative sequence lengths ``(N_seqs + 1,)`` for packed mode.
            When provided, recurrent state is reset at each sequence boundary.

        Returns
        -------
        Tensor
            A tensor of shape (B, T, H) or (B, T, 2*H) containing the hidden
            states at each time step (or just the last state when
            ``last_state_only=True``).
        """
        if h_prev is None:
            h_prev = self._init_hidden_state(x.shape[0], x.device)

        if last_state_only:
           return self._compute_last_state(x, h_prev)

        fwd = self._forward_scan(x, h_prev, cu_seqlens=cu_seqlens, attention_mask=attention_mask)

        if not bidirectional:
            return fwd

        bwd = self._backward_scan(x, cu_seqlens=cu_seqlens, attention_mask=attention_mask)

        if bid_merge == "cat":
            return torch.cat([fwd, bwd], dim=-1)
        elif bid_merge == "sum":
            return fwd + bwd
        else:
            raise ValueError(
                f"Unknown bid_merge mode '{bid_merge}'. Use 'cat' or 'sum'."
            )
