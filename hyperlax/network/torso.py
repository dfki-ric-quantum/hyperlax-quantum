from collections.abc import Sequence

import chex
import numpy as np
from flax import linen as nn
from flax.linen.initializers import Initializer, orthogonal

from hyperlax.network.utils import parse_activation_fn


class MLPTorso(nn.Module):
    """MLP torso."""

    layer_sizes: Sequence[int]
    activation: str = "relu"
    use_layer_norm: bool = False
    kernel_init: Initializer = orthogonal(np.sqrt(2.0))

    @nn.compact
    def __call__(self, observation: chex.Array) -> chex.Array:
        """Forward pass."""
        x = observation
        for layer_size in self.layer_sizes:
            x = nn.Dense(layer_size, kernel_init=self.kernel_init)(x)
            if self.use_layer_norm:
                x = nn.LayerNorm(use_scale=False)(x)
            x = parse_activation_fn(self.activation)(x)
        return x


class CNNTorso(nn.Module):
    """2D CNN torso. Expects input of shape (batch, height, width, channels).
    After this torso, the output is flattened."""

    channel_sizes: Sequence[int]
    kernel_sizes: Sequence[int]
    strides: Sequence[int]
    activation: str = "relu"
    use_layer_norm: bool = False
    kernel_init: Initializer = orthogonal(np.sqrt(2.0))

    @nn.compact
    def __call__(self, observation: chex.Array) -> chex.Array:
        """Forward pass."""
        x = observation
        for channel, kernel, stride in zip(
            self.channel_sizes, self.kernel_sizes, self.strides, strict=False
        ):
            x = nn.Conv(channel, (kernel, kernel), (stride, stride))(x)
            if self.use_layer_norm:
                x = nn.LayerNorm(use_scale=False)(x)
            x = parse_activation_fn(self.activation)(x)

        return x.reshape(*observation.shape[:-3], -1)
