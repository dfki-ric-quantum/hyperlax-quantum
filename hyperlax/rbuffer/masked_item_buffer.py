"""Masked item-based replay buffer supporting vectorization over hyperparameter choices like sample batch size, buffer capacity, etc."""

import functools
from collections.abc import Callable
from typing import Generic, TypeVar

import jax
import jax.numpy as jnp
from chex import Array, ArrayTree, PRNGKey, dataclass

Experience = TypeVar("Experience", bound=ArrayTree)


@dataclass(frozen=True)
class MaskedItemBufferState(Generic[Experience]):
    """State tracking for masked buffer."""

    experience: Experience  # pytree
    mask: Array  # shape: [parallel_envs, buffer_size_per_env], dtype: bool
    current_index: Array  # scalar integer
    is_full: Array  # scalar boolean
    effective_buffer_size_per_env: Array  # scalar integer - max size for this config


@dataclass(frozen=True)
class MaskedItemBufferSample(Generic[Experience]):
    """Container for samples from the masked buffer."""

    experience: Experience


def init(
    experience: Experience,
    parallel_envs: int,
    buffer_size_per_env: int,
    effective_buffer_size_per_env: int,
) -> MaskedItemBufferState[Experience]:
    """Initialize buffer state for a single configuration with empty experience arrays"""
    experience = jax.tree_util.tree_map(jnp.empty_like, experience)
    experience = jax.tree_util.tree_map(
        lambda x: jnp.broadcast_to(
            x[None, None, ...], (parallel_envs, buffer_size_per_env, *x.shape)
        ),
        experience,
    )
    # Initialize mask array
    mask = jnp.zeros((parallel_envs, buffer_size_per_env), dtype=jnp.float32)
    return MaskedItemBufferState(
        experience=experience,
        mask=mask,
        current_index=jnp.array(0, dtype=jnp.int32),
        is_full=jnp.array(False, dtype=bool),
        effective_buffer_size_per_env=jnp.array(effective_buffer_size_per_env, dtype=jnp.int32),
    )


def add(
    state: MaskedItemBufferState[Experience],
    batch: Experience,
    transition_mask: Array,
) -> MaskedItemBufferState[Experience]:
    """Add experiences with masks to buffer."""
    # Get sequence length from batch
    seq_len = jax.tree_util.tree_leaves(batch)[0].shape[1]
    # Calculate indices for inserting new batch
    indices = (
        jnp.arange(seq_len, dtype=jnp.int32) + state.current_index
    ) % state.effective_buffer_size_per_env
    # Update experience and mask arrays
    experience = jax.tree_util.tree_map(
        lambda exp_field, batch_field: exp_field.at[:, indices].set(batch_field),
        state.experience,
        batch,
    )
    new_mask = state.mask.at[:, indices].set(transition_mask)
    # Update buffer state
    new_index = (state.current_index + seq_len).astype(jnp.int32)
    is_full = state.is_full | (new_index >= state.effective_buffer_size_per_env)
    new_index = new_index % state.effective_buffer_size_per_env
    return state.replace(
        experience=experience, mask=new_mask, current_index=new_index, is_full=is_full
    )


def sample(
    state: MaskedItemBufferState[Experience],
    rng_key: PRNGKey,
    batch_size: int,
) -> MaskedItemBufferSample[Experience]:
    """Sample valid experiences from buffer."""
    actual_buffer_size_per_env = state.mask.shape[1]
    # Reshape mask to combine parallel envs and buffer size per env
    valid_mask = state.mask.reshape(-1)
    # Create a probability distribution based on the mask
    # This will be zero for invalid positions and uniform for valid positions
    # (NOTE simpler than sorting indices based on valid and probably more efficient as well?)
    probs = valid_mask.astype(jnp.float32)
    sum_probs = jnp.sum(probs) + 1e-10  # Avoid division by zero
    probs = probs / sum_probs
    # Sample indices according to this probability distribution
    # This will only sample from valid positions if any exist
    rand_indices = jax.random.choice(
        rng_key,
        jnp.arange(valid_mask.shape[0]),
        shape=(batch_size,),
        p=probs,
        replace=True,  # Always sample with replacement to ensure batch_size samples
    )
    # Convert flat indices to 2D indices
    env_idx = (rand_indices // actual_buffer_size_per_env).astype(jnp.int32)
    time_idx = (rand_indices % actual_buffer_size_per_env).astype(jnp.int32)
    # Gather sampled experiences
    sampled_experience = jax.tree_util.tree_map(lambda x: x[env_idx, time_idx], state.experience)
    return MaskedItemBufferSample(experience=sampled_experience)


@dataclass(frozen=True)
class MaskedItemBufferFnContainer(Generic[Experience]):
    """Buffer container with masked operations."""

    init: Callable
    add: Callable
    sample: Callable


def build_masked_item_buffer(
    max_parallel_envs: int,
    buffer_size_per_env: int,
    effective_buffer_size_per_env: int,
    sample_batch_size: int,
) -> MaskedItemBufferFnContainer:
    """Create a masked item buffer for a single configuration."""
    init_fn = functools.partial(
        init,
        parallel_envs=max_parallel_envs,
        buffer_size_per_env=buffer_size_per_env,
        effective_buffer_size_per_env=effective_buffer_size_per_env,
    )
    add_fn = add
    sample_fn = functools.partial(sample, batch_size=sample_batch_size)
    return MaskedItemBufferFnContainer(init=init_fn, add=add_fn, sample=sample_fn)
