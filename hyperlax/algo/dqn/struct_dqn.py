from typing import Any, NamedTuple

import chex
import jax.numpy as jnp
from jumanji.types import TimeStep

from hyperlax.base_types import LogEnvState, OnlineAndTarget, OptStates
from hyperlax.hyperparam.batch import HyperparamBatch
from hyperlax.normalizer.running_stats import NormParams


class DQNMaskMaxDimensions(NamedTuple):
    num_envs: int
    rollout_length: int
    epochs: int
    batch_size: int
    buffer_size: int
    warmup_rollout_length: int


def get_dqn_max_masked_dims(
    batch: HyperparamBatch, num_devices: int, update_batch_size: int
) -> DQNMaskMaxDimensions:
    """
    Returns the max of masked dimensions as a NamedTuple,
    calculating max environments per device based on total_num_envs.
    """
    max_total_envs_global = int(jnp.max(batch.total_num_envs))

    cores_for_dist = num_devices * update_batch_size
    if cores_for_dist == 0:
        cores_for_dist = 1

    max_num_envs_per_core = max_total_envs_global // cores_for_dist

    return DQNMaskMaxDimensions(
        num_envs=max_num_envs_per_core,
        rollout_length=int(jnp.max(batch.rollout_length)),
        epochs=int(jnp.max(batch.epochs)),
        batch_size=int(jnp.max(batch.total_batch_size)),
        buffer_size=int(jnp.max(batch.total_buffer_size)),
        warmup_rollout_length=int(jnp.max(batch.warmup_rollout_length)),
    )


class DQNTransition(NamedTuple):
    obs: chex.ArrayTree
    action: chex.Array
    reward: chex.Array
    done: chex.Array
    next_obs: chex.ArrayTree
    info: dict


class DQNNonVecHyperparams(NamedTuple):
    use_double_q: bool


class DQNVectorizedHyperparams(NamedTuple):
    critic_lr: chex.Array
    tau: chex.Array
    gamma: chex.Array
    max_grad_norm: chex.Array
    training_epsilon: chex.Array
    evaluation_epsilon: chex.Array
    max_abs_reward: chex.Array
    huber_loss_parameter: chex.Array
    warmup_rollout_length: chex.Array
    total_num_envs: chex.Array
    rollout_length: chex.Array
    epochs: chex.Array
    total_buffer_size: chex.Array
    total_batch_size: chex.Array
    decay_learning_rates: chex.Array
    normalize_observations: chex.Array
    sample_id: chex.Array


class DQNLearnerState(NamedTuple):
    params: OnlineAndTarget
    opt_states: OptStates
    buffer_state: Any
    key: chex.PRNGKey
    env_state: LogEnvState
    timestep: TimeStep
    algo_hyperparams: DQNVectorizedHyperparams
    normalization_params: NormParams
    total_env_steps_counter: chex.Array
