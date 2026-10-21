"""Utilities for matrix and kernel initialization."""

__all__ = [
    "init_bias",
    "init_conv1d",
    "init_diagonal_kernel",
    "init_multihead_diagonal_kernel",
    "init_kernel",
]

import random

import numpy as np
import torch
from torch import Tensor


def init_kernel(
    M: int,
    N: int,
    initialization: str = "gamma",
    scaling: float | None = None,
    lambdas: Tensor | None = None,
) -> Tensor:
    """Initialize a (`M`, `N`) complex-valued fully-connected matrix.

    Parameters
    ----------
    M : int
        Number of rows.
    N : int
        Number of columns.
    initialization : str, optional, default='gamma'
        Initialization method. Options are 'gamma' and 'uniform'. 
    scaling : float, optional, default=None
        Scaling factor. Used only if `initialization` is 'uniform'. If 
        'gamma', the matrix is scaled proportionaly to sqrt(1 - |lambda_i|^2), where 
        lambda_i are the eigenvalues of the reservoir weight matrix
    lambdas: Optional[Tensor], optional
        Diagonal elements (eigenvalues) of the recurrent kernel. Only used if scaling
        is 'gamma'.

    Returns
    -------
    Tensor
        A (`M`, `N`) complex-valued tensor containing the initialized fully-connected
        matrix.
    """
    W = torch.zeros((M, N), dtype=torch.complex64).uniform_(-1, 1)

    if initialization == 'gamma': # scale by sqrt(1 - |lambda_i|^2)
        scaling = torch.sqrt(1 - torch.abs(lambdas) ** 2).reshape(1, -1)  # (N, 1)
        W *= scaling
    elif initialization == 'uniform': # scale by given factor
        W *= scaling / torch.linalg.norm(W)
    else:
        raise ValueError(
            f"Unknown initialization method '{initialization}'. "
            "Supported methods are 'gamma' and 'uniform'."
        )

    return W


def init_diagonal_kernel(
    M: int,
    leaky: float,
    rho_min: float = 0.0,
    rho_max: float = 1.0,
    phase_min: float = 0.0,
    phase_max: float = 2 * np.pi,
) -> Tensor:
    """Initialize a (`M`, `M`) complex-valued diagonal matrix.

    The diagonal elements are sampled in a desired subspace of the complex plane, which
    makes it easy to control the eigenvalues of the matrix. Then, to account for
    leakage, the center of the eigenvalues is shifted. The matrix is stored as a 1D
    vector of shape (`M`,), as we are only interested in the non-zero, diagonal
    elements.

    Parameters
    ----------
    M : int
        Matrix size.
    leaky : float
        Leaky rate, which is used to shift the center of the eigenvalues.
    rho_min : float, optional
        Desired minimum radius (magnitude) of the eigenvalues. Should be smaller or
        equal than `rho_max`.
    rho_max : float, optional
        Desired maximum radius (magnitude) of the eigenvalues. Should be greater or
        equal than `rho_min`.
    phase_min : float, optional
        Desired minimum phase (angle) of the eigenvalues. Should be smaller or equal
        than `phase_max`.
    phase_max : float, optional
        Desired maximum phase (angle) of the eigenvalues. Should be greater or equal
        than `phase_min`.

    Returns
    -------
    Tensor
        A (`M`,) complex-valued Tensor containing the diagonal elements of the matrix.
    """
    if rho_min > rho_max:
        print(
            f"rho_min ({rho_min}) should be smaller or equal than rho_max ({rho_max}). "
            "Defaulting to rho_min = rho_max."
        )
        rho_min = rho_max
    if phase_min > phase_max:
        print(
            f"phase_min ({phase_min}) should be smaller or equal than phase_max "
            f"({phase_max}). Defaulting to phase_min = phase_max."
        )
        phase_min = phase_max

    # radii (magnitudes)
    radii = torch.sqrt(torch.zeros(M).uniform_(rho_min**2, rho_max**2))
    # phases (angles)
    thetas = torch.zeros(M).uniform_(phase_min, phase_max)

    return (1 - leaky) + leaky * (radii * torch.exp(1j * thetas))


def init_conv1d(
    shape: tuple[int, ...],
    scaling: float = 1.0,
) -> Tensor:
    """
    Initialize a (input_channels (`i_c`), output_channels (`o_c`), kernel_size (`k`))
    complex-valued 1D convolutional kernel.

    The kernel is initialized uniformly in the range [-1, 1] and then scaled by the
    specified `scaling` factor. Additionally, the kernel is normalized to have a unit
    norm, which is useful for ensuring that the convolution operation does not
    disproportionately amplify or attenuate the input signal.

    Parameters
    ----------
    shape: tuple[int, ...],
        Shape of the kernel to be initialized.
    scaling : float, optional
        Scaling factor for the kernel weights. Default is 1.0.

    Args:
        i_c: Number of input channels.
        o_c: Number of output channels.
        k: Kernel size.
        scaling: Scaling factor.

    Returns
    -------
    Tensor
        A (`i_c`, `o_c`, `k`) tensor representing the initialized convolutional kernel.
    """
    K = torch.zeros(shape, dtype=torch.complex64).uniform_(-1, 1)
    K *= scaling / torch.linalg.norm(K)
    return K


def init_bias(
    M: int,
    scaling: float = 1.0,
) -> Tensor:
    """Initialize a (`M`,) complex-valued bias vector.

    The bias vector is initialized uniformly in the range [-1, 1] and then scaled by the
    specified `scaling` factor.

    Parameters
    ----------
    M : int
        Size of the bias vector.
    scaling : float, optional
        Scaling factor for the bias. Default is 1.0.

    Returns
    -------
    Tensor
        A (`M`,) complex-valued tensor containing the initialized bias vector.
    """
    return torch.zeros(M, dtype=torch.complex64).uniform_(-scaling, scaling)


def init_multihead_diagonal_kernel(
    M: int,
    n_blocks: int,
    leaky: float,
    rho_min: float = 0.0,
    rho_max: float = 1.0,
    phase_min: float = 0.0,
    phase_max: float = 2 * np.pi,
) -> Tensor:
    """Initialize a multi-head diagonal kernel with diverse dynamics per block.

    The ``M`` diagonal elements are partitioned into ``n_blocks`` contiguous
    blocks ("heads").  Each block is initialized with a distinct sub-range of
    spectral radii and phases, linearly interpolated from the global
    ``[rho_min, rho_max]`` and ``[phase_min, phase_max]`` intervals.

    Head 0 captures fast / short-term dynamics (lower ρ), while head K-1
    captures slow / long-term dynamics (higher ρ), mimicking the diversity
    of Multi-Head Attention.

    Parameters
    ----------
    M : int
        Total number of diagonal elements (hidden units).
    n_blocks : int
        Number of heads / blocks to partition the diagonal into.
    leaky : float
        Leaky rate used to shift the centre of eigenvalues.
    rho_min : float, optional
        Global minimum spectral radius.
    rho_max : float, optional
        Global maximum spectral radius.
    phase_min : float, optional
        Global minimum phase (angle).
    phase_max : float, optional
        Global maximum phase (angle).

    Returns
    -------
    Tensor
        A (``M``,) complex-valued tensor with block-wise diverse dynamics.
    """
    if n_blocks <= 0:
        raise ValueError(f"n_blocks must be > 0, got {n_blocks}")
    if n_blocks > M:
        raise ValueError(
            f"n_blocks ({n_blocks}) must be <= M ({M})"
        )

    block_size = M // n_blocks
    remainder = M % n_blocks

    rho_step = (rho_max - rho_min) / n_blocks
    phase_step = (phase_max - phase_min) / n_blocks

    blocks = []
    for k in range(n_blocks):
        size_k = block_size + (remainder if k == n_blocks - 1 else 0)

        # 1. Calcoliamo lo spostamento casuale (jitter)
        # Genera un valore tra -0.3 e +0.3 del passo
        rho_jitter = random.uniform(-0.3, 0.3) * rho_step
        phase_jitter = random.uniform(-0.3, 0.3) * phase_step

        # 2. Applichiamo il jitter (ma non al primo blocco per preservare il minimo assoluto)
        if k > 0:
            rho_min_k = rho_min + k * rho_step + rho_jitter
            phase_min_k = phase_min + k * phase_step + phase_jitter
        else:
            rho_min_k = rho_min
            phase_min_k = phase_min

        # Gestione del massimo per la fase (anche qui aggiungiamo il jitter se non è l'ultimo)
        if k < n_blocks - 1:
            # Il max del blocco k deve idealmente coincidere con il min del blocco k+1 indotto dal jitter successivo
            # Per semplicità ricalcoliamo il punto successivo con un nuovo jitter coerente o lo stimiamo:
            next_phase_jitter = random.uniform(-0.3, 0.3) * phase_step

            phase_max_k = phase_min + (k + 1) * phase_step + next_phase_jitter
        else:
            phase_max_k = phase_min + (k + 1) * phase_step

        # Nel tuo codice rho_max_k è fisso a rho_max, lo manteniamo così
        rho_max_k = rho_max

        # print(f"N. Block {k}: rho_min: {rho_min_k:.2f}, rho_max: {rho_max_k:.4f}")

        block_diag = init_diagonal_kernel(
            M=size_k,
            leaky=leaky,
            rho_min=rho_min_k,
            rho_max=rho_max_k,
            phase_min=phase_min_k,
            phase_max=phase_max_k,
        )
        blocks.append(block_diag)
    # print("\n")

    return torch.cat(blocks, dim=0)

