from typing import Any, NamedTuple

import chex
import jax.numpy as jnp
from jumanji.types import TimeStep

from hyperlax.base_types import LogEnvState, OnlineAndTarget, OptStates
from hyperlax.hyperparam.batch import HyperparamBatch
from hyperlax.normalizer.running_stats import NormParams


class SACTransition(NamedTuple):
    obs: chex.ArrayTree
    action: chex.Array
    reward: chex.Array
    done: chex.Array
    next_obs: chex.ArrayTree
    info: dict

class SACParams(NamedTuple):
    actor_params: chex.ArrayTree
    q_params: OnlineAndTarget
    log_alpha: chex.Array


class SACOptStates(NamedTuple):
    actor_opt_state: OptStates
    q_opt_state: OptStates
    alpha_opt_state: OptStates


class SACNonVecHyperparams(NamedTuple):
    """Container for non-vectorized SAC hyperparameters."""

    autotune: bool


class SACVectorizedHyperparams(NamedTuple):
    """Vectorized hyperparameters for SAC, passed through JAX transformations."""

    actor_lr: chex.Array
    q_lr: chex.Array
    alpha_lr: chex.Array
    gamma: chex.Array
    tau: chex.Array
    max_grad_norm: chex.Array
    target_entropy_scale: chex.Array
    init_alpha: chex.Array
    total_num_envs: chex.Array
    rollout_length: chex.Array
    warmup_rollout_length: chex.Array
    epochs: chex.Array
    total_buffer_size: chex.Array
    total_batch_size: chex.Array
    normalize_observations: chex.Array
    sample_id: chex.Array


class SACLearnerState(NamedTuple):
    """State of the SAC learner with decoupled hyperparams."""

    params: SACParams
    opt_states: SACOptStates
    buffer_state: Any
    key: chex.PRNGKey
    env_state: LogEnvState
    timestep: TimeStep
    algo_hyperparams: SACVectorizedHyperparams
    normalization_params: NormParams
    total_env_steps_counter: chex.Array


class SACMaskMaxDimensions(NamedTuple):
    num_envs: int
    rollout_length: int
    epochs: int
    batch_size: int
    buffer_size: int
    warmup_rollout_length: int


def get_sac_max_masked_dims(
    batch: HyperparamBatch, num_devices: int, update_batch_size: int
) -> SACMaskMaxDimensions:
    """
    Returns the max of masked dimensions as a NamedTuple,
    calculating max environments per device based on total_num_envs.
    """
    max_total_envs_global = int(jnp.max(batch.total_num_envs))
    cores_for_dist = num_devices * update_batch_size
    if cores_for_dist == 0:
        cores_for_dist = 1

    max_num_envs_per_core = max_total_envs_global // cores_for_dist

    return SACMaskMaxDimensions(
        num_envs=max_num_envs_per_core,
        rollout_length=int(jnp.max(batch.rollout_length)),
        epochs=int(jnp.max(batch.epochs)),
        batch_size=int(jnp.max(batch.total_batch_size)),
        buffer_size=int(jnp.max(batch.total_buffer_size)),
        warmup_rollout_length=int(jnp.max(batch.warmup_rollout_length)),
    )
