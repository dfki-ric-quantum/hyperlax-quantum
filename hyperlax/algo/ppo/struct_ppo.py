from typing import NamedTuple

import chex
import jax.numpy as jnp
from jumanji.types import TimeStep

from hyperlax.base_types import (
    ActorCriticOptStates,
    ActorCriticParams,
    LogEnvState,
    Observation,
)
from hyperlax.hyperparam.batch import HyperparamBatch
from hyperlax.network.hyperparam import MLPVectorizedHyperparams
from hyperlax.normalizer.running_stats import NormParams


class PPONonVecHyperparams(NamedTuple):
    """Container for non-vectorized PPO hyperparameters."""

    # For PPO, all hyperparameters are currently vectorized.
    # This class exists to satisfy the AlgorithmInterface.
    pass


class PPOMaxMaskedDims(NamedTuple):
    envs: int
    rollout: int
    epochs: int
    minibatches: int


def get_ppo_max_masked_dims(
    batch: HyperparamBatch, num_devices: int, update_batch_size: int
) -> PPOMaxMaskedDims:
    """
    Returns the max of masked dimensions as a NamedTuple,
    calculating max environments per device based on total_num_envs.
    """
    max_total_envs_global = int(jnp.max(batch.total_num_envs))

    cores_for_dist = num_devices * update_batch_size
    if cores_for_dist == 0:
        cores_for_dist = 1  # Avoid division by zero

    max_num_envs_per_core = max_total_envs_global // cores_for_dist

    return PPOMaxMaskedDims(
        envs=max_num_envs_per_core,
        rollout=int(jnp.max(batch.rollout_length)),
        epochs=int(jnp.max(batch.epochs)),
        minibatches=int(jnp.max(batch.num_minibatches)),
    )


class PPOVectorizedHyperparams(NamedTuple):
    """Vectorized hyperparameters for PPO, passed through JAX transformations."""

    actor_lr: chex.Array
    critic_lr: chex.Array
    gamma: chex.Array
    gae_lambda: chex.Array
    clip_eps: chex.Array
    ent_coef: chex.Array
    vf_coef: chex.Array
    max_grad_norm: chex.Array
    rollout_length: chex.Array
    epochs: chex.Array
    num_minibatches: chex.Array
    total_num_envs: chex.Array
    standardize_advantages: chex.Array
    decay_learning_rates: chex.Array
    normalize_observations: chex.Array
    sample_id: chex.Array


class PPOOnPolicyLearnerState(NamedTuple):
    """State of the PPO learner with decoupled hyperparams."""

    params: ActorCriticParams
    opt_states: ActorCriticOptStates
    key: chex.PRNGKey
    env_state: LogEnvState
    timestep: TimeStep
    algo_hyperparams: PPOVectorizedHyperparams
    # This can be None if the network is not parametric
    actor_network_hyperparams: MLPVectorizedHyperparams | None
    critic_network_hyperparams: MLPVectorizedHyperparams | None
    normalization_params: NormParams
    total_env_steps_counter: chex.Array


class PPOTransition(NamedTuple):
    done: chex.Array
    truncated: chex.Array
    action: chex.Array
    value: chex.Array
    reward: chex.Array
    log_prob: chex.Array
    obs: Observation
    info: dict
