"""Implementation of Linear Echo State Networks (LESN) in PyTorch."""

__all__ = ["LinearESN", "get_linearesn_"]


from typing import Optional
import torch
import torch.nn as nn
from torch import Tensor

from paralesn.layers import (
    ConvFeatureMixer,
    ConvFeatureMixerConfig,
    FullyConnFeatureMixer,
    FullyConnMixerConfig,
    LinearReservoir,
    LinearReservoirConfig,
)


class LinearESNLayer(nn.Module):
    """Implementation of a (linear reservoir -> feature mixer) layer of LinearESN.

    Parameters
    ----------
    reservoir : LinearReservoir
        An instance of the LinearReservoir() class.
    mixer : FeatureMixer
        An instance of the FeatureMixer() class.
    bidirectional : bool
        Whether the reservoir should run in bidirectional mode.
    bid_merge : str
        How to merge forward/backward states (``"cat"`` or ``"sum"``).
    """

    def __init__(
        self,
        reservoir: LinearReservoir,
        mixer: FullyConnFeatureMixer | ConvFeatureMixer,
        bidirectional: bool = False,
        bid_merge: str = "cat",
    ) -> None:
        super().__init__()
        assert isinstance(reservoir, LinearReservoir), (
            "`reservoir` must be an instance of `LinearReservoir`."
        )
        assert isinstance(mixer, (FullyConnFeatureMixer, ConvFeatureMixer)), (
            "`mixer` must be an instance of `FullyConnFeatureMixer` or `ConvFeatureMixer`."
        )

        self.reservoir = reservoir
        self.mixer = mixer
        self.bidirectional = bidirectional
        self.bid_merge = bid_merge

    def forward(
        self, 
        x: Tensor, 
        last_state_only: bool = False,
        cu_seqlens: Tensor | None = None,
        attention_mask: Tensor | None = None,
    ) -> Tensor:
        """Forward pass through the linear reservoir and the feature mixer."""
        return self.mixer(
            self.reservoir(
                x, 
                last_state_only=last_state_only,
                bidirectional=self.bidirectional,
                bid_merge=self.bid_merge,
                cu_seqlens=cu_seqlens,
                attention_mask=attention_mask,
            ),
            attention_mask=attention_mask,
        )


class LinearESN(nn.Module):
    """Linear Reservoir Echo State Network (LinearESN).

    LinearESN is heavily influenced by Linear Recurrent Unit (LRU) [1]_. It consists of
    a linear reservoir, without non-linearity, and a mixing layer. The recurrent kernel
    is a complex-valued, diagonal matrix. Multiple pairs of (linear reservoir, mixing
    layer) may be stacked on top of each other to achieved a DeepLinESN.

    Parameters
    ----------
    in_size : int, optional
        Number of expected input features.
    n_units : int, optional
        Number of recurrent neurons in the reservoir(s).
    n_layers : int, optional
        Number of (linear reservoir -> feature mixer) layers.
    concat : bool, optional
        Whether to concatenate the hidden states of all layers or return only the
        last layer's hidden states.
    bidirectional : bool, optional
        Whether to run the reservoir in bidirectional mode (forward + backward
        scan).
    bid_merge : str, optional
        Merge strategy for bidirectional states.  ``"cat"`` concatenates along the
        hidden dimension (doubling it); ``"sum"`` adds element-wise (preserving
        dimension).
    reservoir_config : LinearReservoirConfig, optional
        Configuration for the first reservoir layer (i.e., l = 1).
    mixer_config : FeatureMixerConfig, optional
        Configuration for the first mixing layer (i.e., l = 1).
    inter_reservoir_config : LinearReservoirConfig, optional
        Configuration for the reservoir layers after the first one (i.e., l > 1).
        If None, the same configuration as `reservoir_config` is used.
    inter_mixer_config : FeatureMixerConfig, optional
        Configuration for the mixing layers after the first one (i.e., l > 1).
        If None, the same configuration as `mixer_config` is used.

    Attributes
    ----------
    first_layer_units : The number of neurons for the first reservoir. Used only if the
    number of layers is grater than 1 and `concat` is True. Otherwise it corresponds to
    `n_units`.

    References
    ----------
    .. [1] Orvieto, Antonio, et al. "Resurrecting recurrent neural networks for long
    sequences." International Conference on Machine Learning. PMLR, 2023.
    """

    def __init__(
        self,
        in_size: int = 1,
        n_units: int = 128,
        model_dim: Optional[int] = None,   
        n_layers: int = 1,
        concat: bool = False,
        bidirectional: bool = False,
        bid_merge: str = "cat",
        reservoir_config: LinearReservoirConfig = None,
        mixer_config: ConvFeatureMixerConfig | FullyConnMixerConfig = None,
        inter_reservoir_config: LinearReservoirConfig | None = None,
        inter_mixer_config: ConvFeatureMixerConfig | FullyConnMixerConfig | None = None,
        **kwargs,
    ) -> None:
        """
        Parameters
        ----------
        in_size : int, optional
            Number of expected input features.
        n_units : int, optional
            Number of recurrent neurons in the reservoir(s).
        n_layers : int, optional
            Number of (linear reservoir -> feature mixer) layers.
        concat : bool, optional
            Whether to concatenate the hidden states of all layers or return only the
            last layer's hidden states.
        bidirectional : bool, optional
            Whether to run in bidirectional mode.
        bid_merge : str, optional
            Merge strategy for bidirectional states (``"cat"`` or ``"sum"``).
        reservoir_config : LinearReservoirConfig, optional
            Configuration for the first reservoir layer (i.e., l = 1).
            Configuration for the first mixing layer (i.e., l = 1).
        inter_reservoir_config : LinearReservoirConfig, optional
            Configuration for the reservoir layers after the first one (i.e., l > 1).
            If None, the same configuration as `reservoir_config` is used.
        inter_mixer_config : FeatureMixerConfig, optional
            Configuration for the mixing layers after the first one (i.e., l > 1).
            If None, the same configuration as `mixer_config` is used.
        """
        super().__init__()
        assert n_layers > 0, "`n_layers` must be greater than 0."

        if reservoir_config is None:
            reservoir_config = LinearReservoirConfig()
        if mixer_config is None:
            mixer_config = ConvFeatureMixerConfig()

        self.__dict__.update(kwargs)
        self.in_size = in_size
        if model_dim is None:
            model_dim = n_units
        self.model_dim = model_dim
        self.n_units = n_units
        self.n_layers = n_layers
        self.concat = concat
        self.bidirectional = bidirectional
        self.bid_merge = bid_merge

        # for the first layer (i.e., l = 1)
        self.reservoir_config = reservoir_config
        self.mixer_config = mixer_config

        # for subsequent layers (i.e., l > 1)
        self.inter_reservoir_config = reservoir_config
        if inter_reservoir_config:
            self.inter_reservoir_config = inter_reservoir_config
        self.inter_mixer_config = mixer_config
        if inter_mixer_config:
            self.inter_mixer_config = inter_mixer_config

        # if concat == True the number of units for the model size is distributed across layers, otherwise each layer has n_units
        self.layers_units = self.first_layer_units = n_units
        self.layers_model_size = self.first_layer_model_size = model_dim
        if concat:
            self.layers_units = n_units // n_layers
            self.first_layer_units = self.layers_units + n_units % n_layers
            self.layers_model_size = model_dim // n_layers
            self.first_layer_model_size = self.layers_model_size + model_dim % n_layers

        self.layers = self._make_layers()

    def _effective_reservoir_out(self, n_units_layer: int) -> int:
        """Return the effective output dimension of a reservoir layer,
        accounting for bidirectional concatenation."""
        if self.bidirectional and self.bid_merge == "cat":
            return 2 * n_units_layer
        return n_units_layer

    def _make_layers(self) -> nn.Sequential:
        """Initialize the (linear reservoir -> feature mixer) layers of LinearESN.

        Returns
        -------
        nn.Sequential
            A sequential container of the (linear reservoir -> feature mixer) layers.
        """
        layers = []

        # --- first layer ---
        first_eff = self._effective_reservoir_out(self.first_layer_units)

        if isinstance(self.mixer_config, ConvFeatureMixerConfig):
            mixer_class = ConvFeatureMixer
        else: # FullyConnFeatureMixerConfig
            self.mixer_config.n_units = first_eff
            self.mixer_config.model_dim = self.first_layer_model_size
            mixer_class = FullyConnFeatureMixer

        layers.append(
            LinearESNLayer(
                reservoir=LinearReservoir(
                    in_size=self.in_size,
                    n_units=self.first_layer_units,
                    config=self.reservoir_config,
                ),
                mixer=mixer_class(
                    config=self.mixer_config,  # type: ignore
                ),
                bidirectional=self.bidirectional,
                bid_merge=self.bid_merge,
            )
        )

        # --- subsequent layers ---
        inter_eff = self._effective_reservoir_out(self.layers_units)

        if isinstance(self.inter_mixer_config, ConvFeatureMixerConfig):
            inter_mixer_class = ConvFeatureMixer
        else: # FullyConnFeatureMixerConfig
            self.inter_mixer_config.n_units = inter_eff
            self.inter_mixer_config.model_dim = self.layers_model_size
            inter_mixer_class = FullyConnFeatureMixer

        # if concat == True the hidden size of the first layer may vary based on the
        # remainder from the units distribution
        h_dim = self.first_layer_model_size
        for _ in range(1, self.n_layers):
            layers.append(
                LinearESNLayer(
                    reservoir=LinearReservoir(
                        in_size=h_dim,
                        n_units=self.layers_units,
                        config=self.inter_reservoir_config,
                    ),
                    mixer=inter_mixer_class(
                        config=self.inter_mixer_config,  # type: ignore
                ),
                bidirectional=self.bidirectional,
                bid_merge=self.bid_merge,
            )
            )
            h_dim = self.layers_model_size

        return nn.Sequential(*layers)

    @torch.no_grad()
    def forward(
        self, 
        x: Tensor, 
        last_state_only: bool = False,
        cu_seqlens: Tensor | None = None,
        attention_mask: Tensor | None = None,
    ) -> Tensor:
        """Forward pass through the LinearESN.

        Parameters
        ----------
        x : Tensor
            External input tensor of shape (batch_size (B), sequence length (T),
            n_features (N)).
        cu_seqlens : Tensor, optional
            Cumulative sequence lengths ``(N_seqs + 1,)`` for packed mode.

        Returns
        -------
        Tensor
            If `concat` is False, a tensor containing the last layer's hidden states,
            of shape (B, 1, hidden_size (H)). If `concat` is True, a tensor containing
            the collection of hidden states from all layers, of shape (B, T, H).
        """
        if self.n_layers == 1:
            x = x.to(torch.complex64)  # cast to complex
            if last_state_only:
                x = self.layers[0](x, last_state_only=True, attention_mask=attention_mask)
            else:
                x = self.layers[0](x, cu_seqlens=cu_seqlens, attention_mask=attention_mask)
            return x

        match self.concat:
            case True:
                states = []
                for layer in self.layers:
                    x = x.to(torch.complex64)  # cast to complex
                    x = layer(x, cu_seqlens=cu_seqlens, attention_mask=attention_mask)
                    states.append(x)
                return torch.cat(states, dim=2)
            case False:
                for layer in self.layers:
                    x = x.to(torch.complex64)
                    x = layer(x, cu_seqlens=cu_seqlens, attention_mask=attention_mask)
                return x

        # """states = []
        # for layer in self.layers:
        #     x = x.to(torch.complex64)  # cast to complex
        #     x = layer(x)
        #     states.append(x)
        #
        # match self.concat:
        #     case True:
        #         states = torch.cat(states, dim=2)
        #     case False:
        #         states = states[-1]
        #
        # return states"""


def get_linearesn_(hparams: dict) -> LinearESN:
    """Create a LinearESN instance from the given hyperparameters.

    Parameters
    ----------
    hparams : dict
        A dictionary containing the hyperparameters for the LinearESN.
    """
    return LinearESN(**hparams)
