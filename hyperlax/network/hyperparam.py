from typing import NamedTuple

import chex
import jax.numpy as jnp

from hyperlax.hyperparam.batch import HyperparamBatch


class MLPVectorizedHyperparams(NamedTuple):
    """Vectorized hyperparameters for an MLP, passed through JAX transformations."""

    num_layers: chex.Array
    width: chex.Array
    activation: chex.Array
    use_layer_norm: chex.Array


class MLPArchitecturalHyperparamBatch(HyperparamBatch):
    """Wrapper for MLP architectural hyperparameter batches, inheriting from generic batch."""

    def __init__(self, data_values: jnp.ndarray, field_name_to_index: dict[str, int]):
        super().__init__(data_values, field_name_to_index, list(MLPVectorizedHyperparams._fields))
