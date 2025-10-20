import logging
import os

import chex
import jax
import jax.numpy as jnp
import numpy as np

logger = logging.getLogger(__name__)


def scale_gradient(g: chex.Array, scale: float = 1) -> chex.Array:
    """Scales the gradient of `g` by `scale` but keeps the original value unchanged."""
    return g * scale + jax.lax.stop_gradient(g) * (1.0 - scale)


def count_parameters(params: chex.ArrayTree) -> int:
    """Counts the number of parameters in a parameter tree."""
    return sum(x.size for x in jax.tree_util.tree_leaves(params))


def ndim_at_least(x: chex.Array, num_dims: chex.Numeric) -> chex.Array:
    """Check if the number of dimensions of `x` is at least `num_dims`."""
    if not (isinstance(x, jax.Array) or isinstance(x, np.ndarray)):
        x = jnp.asarray(x)
    return x.ndim >= num_dims


def merge_leading_dims(x: chex.Array, num_dims: chex.Numeric) -> chex.Array:
    """Merge leading dimensions.

    Note:
        This implementation is a generic function for merging leading dimensions
        extracted from Haiku.
        For the original implementation, please refer to the following link:
        (https://github.com/deepmind/dm-haiku/blob/main/haiku/_src/basic.py#L207)
    """
    # Don't merge if there aren't dimensions to merge.
    if not ndim_at_least(x, num_dims):
        return x

    new_shape = (np.prod(x.shape[:num_dims]),) + x.shape[num_dims:]
    return x.reshape(new_shape)


def unreplicate_n_dims(x: chex.ArrayTree, unreplicate_depth: int = 2) -> chex.ArrayTree:
    """Unreplicates a pytree by removing the first `unreplicate_depth` axes.

    This function takes a pytree and removes some number of axes, associated with parameter
    duplication for running multiple updates across devices and in parallel with `vmap`.
    This is typically one axis for device replication, and one for the `update batch size`.
    """
    return jax.tree_util.tree_map(lambda x: x[(0,) * unreplicate_depth], x)  # type: ignore


def unreplicate_batch_dim(x: chex.ArrayTree) -> chex.ArrayTree:
    """Unreplicated just the update batch dimension.
    (The dimension that is vmapped over when acting and learning)

    In hyperlax's case it is always the second dimension, after the device dimension.
    We simply take element 0 as the params are identical across this dimension.
    """
    return jax.tree_util.tree_map(lambda x: x[:, 0, ...], x)  # type: ignore


def unreplicate_batch_dim_when_agentseeds_and_hyp(x: chex.ArrayTree) -> chex.ArrayTree:
    """Unreplicated just the update batch dimension.
    (The dimension that is vmapped over when acting and learning)

    In hyperlax's case it is always the second (+hyp_batches) = 4th dimension, after the device dimension.
    We simply take element 0 as the params are identical across this dimension.
    """
    return jax.tree_util.tree_map(lambda x: x[:, :, :, 0, ...], x)  # type: ignore


def print_xla_env_vars():
    logger.info("--- XLA/JAX Environment Variables ---")
    for key, value in os.environ.items():
        if key.startswith("XLA_") or key.startswith("JAX_"):
            logger.info(f"{key}={value}")
    logger.info("------------------------------------")


from pprint import pprint


def print_pytree_shapes(pytree, name=""):
    """Print shapes of all leaves in a pytree."""

    # print(f"\n{name} shapes:")
    def get_shape(x):
        return x.shape if hasattr(x, "shape") else "scalar"

    shapes = jax.tree_util.tree_map(get_shape, pytree)
    pprint(shapes)


def get_pytree_shapes(pytree):
    def get_shape(x):
        return x.shape if hasattr(x, "shape") else "scalar"

    shapes = jax.tree_util.tree_map(get_shape, pytree)
    return shapes
