# NOTE RunningStats with Standard/MeanStd method was largely taken from brax/acme

"""Utility functions to compute running statistics.

This file was taken from acme and modified to simplify dependencies:

https://github.com/deepmind/acme/blob/master/acme/jax/running_statistics.py
"""

import logging
from collections.abc import Callable
from typing import Any, Union

import chex
import jax
import jax.numpy as jnp
from brax.training.acme import specs, types
from flax import struct
from flax.core.frozen_dict import FrozenDict

logger = logging.getLogger(__name__)

NormalizeFn = Callable[
    [FrozenDict, chex.Array], chex.Array
]  # [normalization_params, observation]->observation
UpdateNormFn = Callable[
    [FrozenDict, chex.Array], chex.Array
]  # [normalization_params, observation]->observation
InitNormFn = Callable[[specs.Array], Any]


@struct.dataclass
class NestedMeanStd:
    """A container for running statistics (mean, std) of possibly nested data."""

    mean: types.Nest
    std: types.Nest


@struct.dataclass
class NestedMinMax:
    """A container for min/max values of possibly nested data."""

    min: types.Nest
    max: types.Nest


@struct.dataclass
class RunningStatsMeanStd(NestedMeanStd):
    """Full state of running statistics computation."""

    count: jnp.ndarray
    summed_variance: types.Nest


NormParams = Union[RunningStatsMeanStd, NestedMinMax]


def _zeros_like(nest: types.Nest, dtype=None) -> types.Nest:
    return jax.tree_util.tree_map(lambda x: jnp.zeros(x.shape, dtype or x.dtype), nest)


def _ones_like(nest: types.Nest, dtype=None) -> types.Nest:
    return jax.tree_util.tree_map(lambda x: jnp.ones(x.shape, dtype or x.dtype), nest)


def init_running_meanstd(nest: types.Nest) -> RunningStatsMeanStd:
    """Initializes the running statistics for the given nested structure."""
    dtype = jnp.float64 if jax.config.jax_enable_x64 else jnp.float32

    return RunningStatsMeanStd(
        count=jnp.zeros((), dtype=dtype),
        mean=_zeros_like(nest, dtype=dtype),
        summed_variance=_zeros_like(nest, dtype=dtype),
        # Initialize with ones to make sure normalization works correctly
        # in the initial state.
        std=_ones_like(nest, dtype=dtype),
    )


def _validate_batch_shapes(
    batch: types.NestedArray,
    reference_sample: types.NestedArray,
    batch_dims: tuple[int, ...],
) -> None:
    """Verifies shapes of the batch leaves against the reference sample.

    Checks that batch dimensions are the same in all leaves in the batch.
    Checks that non-batch dimensions for all leaves in the batch are the same
    as in the reference sample.

    Arguments:
        batch: the nested batch of data to be verified.
        reference_sample: the nested array to check non-batch dimensions.
        batch_dims: a Tuple of indices of batch dimensions in the batch shape.

    Returns:
        None.
    """

    def validate_node_shape(reference_sample: jnp.ndarray, batch: jnp.ndarray) -> None:
        expected_shape = batch_dims + reference_sample.shape
        assert batch.shape == expected_shape, f"{batch.shape} != {expected_shape}"

    jax.tree_util.tree_map(validate_node_shape, reference_sample, batch)


def update_running_meanstd(
    state: RunningStatsMeanStd,
    batch: types.Nest,
    *,
    weights: jnp.ndarray | None = None,
    std_min_value: float = 1e-6,
    std_max_value: float = 1e5,
    pmap_axis_name: str | None = None,
    validate_shapes: bool = True,
) -> RunningStatsMeanStd:
    """Updates the running statistics with the given batch of data.

    Note: data batch and state elements (mean, etc.) must have the same structure.

    Note: by default will use int32 for counts and float32 for accumulated
    variance. This results in an integer overflow after 2^31 data points and
    degrading precision after 2^24 batch updates or even earlier if variance
    updates have large dynamic range.
    To improve precision, consider setting jax_enable_x64 to True, see
    https://jax.readthedocs.io/en/latest/notebooks/Common_Gotchas_in_JAX.html#double-64bit-precision

    Arguments:
        state: The running statistics before the update.
        batch: The data to be used to update the running statistics.
        weights: Weights of the batch data. Should match the batch dimensions.
            Passing a weight of 2. should be equivalent to updating on the
            corresponding data point twice.
        std_min_value: Minimum value for the standard deviation.
        std_max_value: Maximum value for the standard deviation.
        pmap_axis_name: Name of the pmapped axis, if any.
        validate_shapes: If true, the shapes of all leaves of the batch will be
            validated. Enabled by default. Doesn't impact performance when jitted.

    Returns:
        Updated running statistics.
    """
    # We require exactly the same structure to avoid issues when flattened
    # batch and state have different order of elements.
    assert jax.tree_util.tree_structure(batch) == jax.tree_util.tree_structure(state.mean)
    batch_shape = jax.tree_util.tree_leaves(batch)[0].shape
    # We assume the batch dimensions always go first.
    batch_dims = batch_shape[: len(batch_shape) - jax.tree_util.tree_leaves(state.mean)[0].ndim]
    batch_axis = range(len(batch_dims))
    if weights is None:
        step_increment = jnp.prod(jnp.array(batch_dims))
    else:
        step_increment = jnp.sum(weights)
    if pmap_axis_name is not None:
        step_increment = jax.lax.psum(step_increment, axis_name=pmap_axis_name)
    count = state.count + step_increment

    # Validation is important. If the shapes don't match exactly, but are
    # compatible, arrays will be silently broadcasted resulting in incorrect
    # statistics.
    if validate_shapes:
        if weights is not None:
            if weights.shape != batch_dims:
                raise ValueError(f"{weights.shape} != {batch_dims}")
        _validate_batch_shapes(batch, state.mean, batch_dims)

    def _compute_node_statistics(
        mean: jnp.ndarray, summed_variance: jnp.ndarray, batch: jnp.ndarray
    ) -> tuple[jnp.ndarray, jnp.ndarray]:
        assert isinstance(mean, jnp.ndarray), type(mean)
        assert isinstance(summed_variance, jnp.ndarray), type(summed_variance)
        # The mean and the sum of past variances are updated with Welford's
        # algorithm using batches (see https://stackoverflow.com/q/56402955).
        diff_to_old_mean = batch - mean
        if weights is not None:
            expanded_weights = jnp.reshape(
                weights, list(weights.shape) + [1] * (batch.ndim - weights.ndim)
            )
            diff_to_old_mean = diff_to_old_mean * expanded_weights
        mean_update = jnp.sum(diff_to_old_mean, axis=batch_axis) / count
        if pmap_axis_name is not None:
            mean_update = jax.lax.psum(mean_update, axis_name=pmap_axis_name)
        mean = mean + mean_update

        diff_to_new_mean = batch - mean
        variance_update = diff_to_old_mean * diff_to_new_mean
        variance_update = jnp.sum(variance_update, axis=batch_axis)
        if pmap_axis_name is not None:
            variance_update = jax.lax.psum(variance_update, axis_name=pmap_axis_name)
        summed_variance = summed_variance + variance_update
        return mean, summed_variance

    updated_stats = jax.tree_util.tree_map(
        _compute_node_statistics, state.mean, state.summed_variance, batch
    )
    # Extract `mean` and `summed_variance` from `updated_stats` nest.
    mean = jax.tree_util.tree_map(lambda _, x: x[0], state.mean, updated_stats)
    summed_variance = jax.tree_util.tree_map(lambda _, x: x[1], state.mean, updated_stats)

    def compute_std(summed_var, prev_std):
        safe_count = jnp.maximum(count, 1.0)
        variance = jnp.maximum(summed_var / safe_count, 0.0)
        computed_std = jnp.sqrt(variance)
        computed_std = jnp.clip(computed_std, std_min_value, std_max_value)
        return jnp.where(count > 0, computed_std, prev_std)

    # Ensure both summed_variance and state.std are compatible
    summed_variance_flat, summed_tree = jax.tree_util.tree_flatten(summed_variance)
    std_flat, std_tree = jax.tree_util.tree_flatten(state.std)

    # Check if the structures match
    assert summed_tree == std_tree, "Structure mismatch between summed_variance and state.std"

    # Compute std for each leaf
    std_flat = [
        compute_std(sv, std) for sv, std in zip(summed_variance_flat, std_flat, strict=False)
    ]
    std = jax.tree_util.tree_unflatten(std_tree, std_flat)

    new_state = RunningStatsMeanStd(
        count=count, mean=mean, summed_variance=summed_variance, std=std
    )

    # Apply mask
    mask = step_increment > 0
    final_state = jax.tree.map(lambda new, old: jnp.where(mask, new, old), new_state, state)

    return final_state


def normalize_meanstd(
    mean_std: NestedMeanStd,
    batch: types.NestedArray,
    max_abs_value: float | None = None,
) -> types.NestedArray:
    """Normalizes data using running statistics."""

    def normalize_leaf(data: jnp.ndarray, mean: jnp.ndarray, std: jnp.ndarray) -> jnp.ndarray:
        # Only normalize inexact
        if not jnp.issubdtype(data.dtype, jnp.inexact):
            return data
        data = (data - mean) / std
        if max_abs_value is not None:
            # TODO: remove pylint directive
            data = jnp.clip(data, -max_abs_value, +max_abs_value)
        return data

    return jax.tree_util.tree_map(normalize_leaf, batch, mean_std.mean, mean_std.std)


def denormalize_meanstd(
    mean_std: NestedMeanStd,
    batch: types.NestedArray,
) -> types.NestedArray:
    """Denormalizes values in a nested structure using the given mean/std.

    Only values of inexact types are denormalized.
    See https://numpy.org/doc/stable/_images/dtype-hierarchy.png for Numpy type
    hierarchy.

    Args:
        batch: a nested structure containing batch of data.
        mean_std: mean and standard deviation used for denormalization.

    Returns:
        Nested structure with denormalized values.
    """

    def denormalize_leaf(data: jnp.ndarray, mean: jnp.ndarray, std: jnp.ndarray) -> jnp.ndarray:
        # Only denormalize inexact
        if not jnp.issubdtype(data.dtype, jnp.inexact):
            return data
        return data * std + mean

    return jax.tree_util.tree_map(denormalize_leaf, batch, mean_std.mean, mean_std.std)


def normalize_minmax(
    # config: DictConfig,
    params: NestedMinMax,
    batch_input: chex.Array,
    epsilon: float = 1e-5,
    feature_range: tuple[float, float] = (-jnp.pi / 2, jnp.pi / 2),
) -> chex.Array:
    def normalize_scale_leaf(
        data: jnp.ndarray,
        min: jnp.ndarray,
        max: jnp.ndarray,
        epsilon: float,
        feature_range: tuple[float, float],
    ) -> jnp.ndarray:
        if not jnp.issubdtype(data.dtype, jnp.inexact):
            return data
        scale = max - min
        scale = jnp.where(scale < epsilon, scale + epsilon, scale)
        data = (data - min) / scale
        data = data * (feature_range[1] - feature_range[0]) + feature_range[0]
        return data

    return jax.tree_util.tree_map(
        normalize_scale_leaf,
        batch_input,
        params.min,
        params.max,
        epsilon,
        feature_range,
    )


def init_norm_identity(nest: types.Nest) -> Any:
    """Identity function for init_state_meanstd."""
    # Convert specs.Array to JAX array if needed
    if isinstance(nest, specs.Array):
        return jnp.zeros(nest.shape, dtype=nest.dtype)
    return jax.tree_util.tree_map(lambda x: jnp.asarray(x), nest)


def update_norm_identity(state: Any, batch: types.Nest) -> Any:
    """Identity function for update_meanstd."""
    return state


def normalize_identity(mean_std: Any, batch: types.NestedArray) -> types.NestedArray:
    """Identity function for normalize_meanstd."""
    return jnp.asarray(batch)


def denormalize_identity(mean_std: Any, batch: types.NestedArray) -> types.NestedArray:
    """Identity function for denormalize_meanstd."""
    return jnp.asarray(batch)


def get_init_fixed_norm_minmax_fn(obs_spec: specs.Array) -> InitNormFn:
    if hasattr(obs_spec.agent_view, "minimum") and hasattr(obs_spec.agent_view, "maximum"):
        if jnp.any(jnp.isinf(obs_spec.agent_view.minimum)) or jnp.any(
            jnp.isinf(obs_spec.agent_view.maximum)
        ):
            raise ValueError(
                "obs_spec.agent_view.minimum is -inf, which means it is unbounded. thus we can't use fixed_minmax normalization"
            )
    else:
        raise ValueError(
            "obs_spec.agent_view doesn't have minimum and maximum attributes. thus we cannot infer the minmax values"
        )

    def init_fixed_norm_minmax_fn(
        nest: jnp.ndarray,
    ) -> NestedMinMax:  # just to have a same signature
        logger.debug("init_fixed_norm_minmax_fn")
        return NestedMinMax(
            min=obs_spec.agent_view.minimum,
            max=obs_spec.agent_view.maximum,
        )

    return init_fixed_norm_minmax_fn


def normalizer_setup(
    normalize_observations: bool, normalize_method: str, obs_spec: specs.Array
) -> tuple[Callable, Callable, Callable]:
    """
    Modified to support weights parameter in update function
    """
    if not isinstance(normalize_observations, bool):
        raise TypeError(
            f"Expected normalize_observations to be bool but got {type(normalize_observations)}"
        )
    if normalize_observations:
        if normalize_method == "running_meanstd":
            logger.info("Setting up normalizer w/ running_meanstd")
            init_fn = init_running_meanstd
            update_fn = lambda params, batch, weights: update_running_meanstd(
                params, batch, weights=weights
            )
            normalize_fn = normalize_meanstd
        elif normalize_method == "fixed_minmax":
            logger.info("Setting up normalizer w/ fixed_minmax")
            init_fn = get_init_fixed_norm_minmax_fn(obs_spec)
            update_fn = lambda params, batch, weights=None: params  # Ignore weights
            normalize_fn = normalize_minmax
        else:
            raise ValueError(f"Unknown normalization method: {normalize_method}")
    else:
        logger.info("Setting up normalizer w/ identity")
        init_fn = init_norm_identity
        update_fn = lambda params, batch, weights=None: params  # Identity update
        normalize_fn = normalize_identity

    # Wrap functions with consistent interface
    def common_init_fn(nest: types.Nest):
        return init_fn(nest)

    def common_update_fn(params: Any, batch: types.Nest, weights: jnp.ndarray | None = None):
        return update_fn(params, batch, weights)

    def common_normalize_fn(params: Any, batch: types.Nest):
        return normalize_fn(params, batch)

    return common_init_fn, common_update_fn, common_normalize_fn
