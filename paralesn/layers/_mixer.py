"""Feature mixing layer for the LinearESN architecture."""

__all__ = ["ConvFeatureMixerConfig", "ConvFeatureMixer", "FullyConnFeatureMixer", "FullyConnMixerConfig"]

from dataclasses import dataclass
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor

from paralesn.utils import init_bias, init_conv1d, init_kernel


@dataclass
class ConvFeatureMixerConfig:
    """Hyperparameters' configuration for the (feature) mixing layer.

    Attributes
    ----------
    activation : Callable[[Tensor], Tensor], optional
        Activation function applied to the output of the mixing layer.
    channels : int, optional
        Number of channels for the mixing layer.
    mix_scaling : float, optional
        Weight scaling for the mixing layer.
    mix_bias_scaling : float, optional
        Bias scaling for the mixing layer.
    kernel_size : Optional[int], optional
        Kernel size for the mixing layer.
    """

    channels: int = 1
    mix_scaling: float = 1.0
    mix_bias_scaling: float = 0.0
    kernel_size: int = 3


class ConvFeatureMixer(nn.Module):
    """Convolutional (feature) mixing layer.

    The mixing layer applies a 1D convolution to the reservoir's output states, to mix
    the features across the hidden dimension. Specifically, a single 1D kernel of shape
    (`config.kernel_size`) is applied to all channels (i.e., the time dimension) of the
    input tensor.

    Parameters
    ----------
    config : MixerConfig
        Configuration for the feature mixing layer.

    Attributes
    ----------
    activation : Callable[[Tensor], Tensor]
        Activation function applied to the output of the mixing layer.
    weight : Tensor
        The 1D kernel of the mixing layer.
    bias : Tensor
        The bias vector of the mixing layer.
    """

    def __init__(self, config: ConvFeatureMixerConfig) -> None:
        """
        Parameters
        ----------
        config : MixerConfig
            Configuration for the feature mixing layer.
        """
        super().__init__()
        self.config = config

        self.activation = nn.Tanh()

        # 1D convolution
        weight = torch.squeeze(
            init_conv1d(
                shape=(config.channels, config.channels, config.kernel_size),
                scaling=config.mix_scaling,
            )
        )
        self.register_buffer("weight", weight)
        bias = init_bias(M=config.channels, scaling=config.mix_bias_scaling)
        self.register_buffer("bias", bias)

    @torch.no_grad()
    def forward(self, x: Tensor, attention_mask: Tensor | None = None) -> Tensor:
        """Forward pass of the mixing layer.

        Parameters
        ----------
        x : Tensor
            Input tensor of shape (batch_size (B), channels (T), hidden_size (H)).  The
            channels are essentially the time dimension/sequence length.

        Returns
        -------
        Tensor
            Output tensor of shape (B, T, H).
        """
        # check whether the input is 2D (only last reservoir state) or 3D (all
        # reservoir states)
        if len(x.shape) == 2:
            x = x.view(x.size(0), 1, x.size(1))
        seqlen = x.shape[1]

        # pad sequence length if needed
        # needed at inference time for regression (i.e., memory and forecasting) tasks,
        # as the sequence length may differ w.r.t. training time
        if seqlen < self.config.channels:
            x = F.pad(
                input=x,
                pad=(0, 0, 0, self.config.channels - seqlen, 0, 0),
                mode="constant",
            )

        # TODO: later on we may want to have a more elegant approach here that doesn't
        # require repeating the weights/biases to all channels but instead just applies
        # self.weights across channels as intended

        # repeat the same kernel of shape (`config.kernel_size`) to math expected
        # sequence length of the input
        weight = self.weight.repeat(seqlen, 1, 1)
        bias = self.bias.repeat(seqlen)

        # with `groups=seqlen` we have independent convolution operations for each
        # input channel (our timesteps). This makes the 1D conv more similar to what
        # happens with the matrix multiplication approach (i.e., in the mixing no
        # looking at previous/future timesteps).
        out = self.activation(
            F.conv1d(
                input=x.real,
                weight=weight.real,
                bias=bias.real,
                groups=seqlen,
                padding="same",
            ) - 
            F.conv1d(
                input=x.imag,
                weight=weight.imag,
                bias=None,
                groups=seqlen,
                padding="same",
            )
        )

        out = out[:, :seqlen, :]
        if attention_mask is not None:
            out = out * attention_mask.unsqueeze(-1)
        return out



@dataclass
class FullyConnMixerConfig:
    n_units: int
    model_dim: int
    mix_scaling: float = 1.0
    mix_bias_scaling: float = 0.0



class FullyConnFeatureMixer(nn.Module):
    def __init__(self, config: FullyConnMixerConfig) -> None:
        """
        Docstring for __init__

        :param self: Description
        :param config: Description
        :type config: FeatureMixerConfig
        """
        super().__init__()
        self.config = config

        self.activation = nn.Tanh()

        weight = init_kernel(M=config.n_units, N=config.model_dim, initialization='uniform', scaling=config.mix_scaling)
        bias = init_bias(M=config.model_dim, scaling=config.mix_bias_scaling)
        self.register_buffer("weight", weight)
        self.register_buffer("bias", bias)

    @torch.no_grad()
    def forward(self, x: Tensor, attention_mask: Tensor | None = None) -> Tensor:
        """Forward pass of the mixing layer.

        Parameters
        ----------
        x : Tensor
            Input tensor of shape (batch_size (B), sequence_length (T), n_units (S)).

        Returns
        -------
        Tensor
            Output tensor of shape (B, T, model_dim (M)).
        """
        out = self.activation(
            F.linear(input=x.real, weight=self.weight.real, bias=self.bias.real) - 
            F.linear(input=x.imag, weight=self.weight.imag, bias=None)
        )

        if attention_mask is not None:
            out = out * attention_mask.unsqueeze(-1)

        return out

