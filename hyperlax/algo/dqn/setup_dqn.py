import logging
from collections.abc import Callable
from typing import Any

import brax.training.acme.specs as brax_specs
import chex
import jax
import jax.numpy as jnp
import optax
from flax.core.frozen_dict import FrozenDict

from hyperlax.algo.dqn.core import (
    DQNLearnerState,
    DQNTransition,
    build_dqn_update_step_fn,
    build_warmup_rollout_fn,
)
from hyperlax.algo.dqn.struct_dqn import (
    DQNVectorizedHyperparams,
    get_dqn_max_masked_dims,
)
from hyperlax.base_types import ActFn, OnlineAndTarget
from hyperlax.configs.main_base import BaseExperimentConfig
from hyperlax.evaluator.core import get_distribution_act_fn
from hyperlax.evaluator.setup import slice_extra_batch_dims
from hyperlax.layout.axes import DistributionStrategy
from hyperlax.layout.data import (
    broadcast_hp_batched_array_to_strategy_shape,
    build_generic_distributed_state,
)
from hyperlax.network.base import FeedForwardActor
from hyperlax.normalizer.running_stats import NormParams
from hyperlax.rbuffer.masked_item_buffer import build_masked_item_buffer
from hyperlax.trainer.phaser import AlgoSetupFns, build_generic_phaser_setup_fns
from hyperlax.utils.algo_setup import (
    AlgorithmNetworkSetup,
    EnvironmentInfo,
    NetworkSpec,
    OptimizerSpec,
)

logger = logging.getLogger(__name__)


def build_dqn_network_setup() -> AlgorithmNetworkSetup:
    """Creates the network and optimizer specification for DQN."""
    return AlgorithmNetworkSetup(
        network_specs={
            "q_network": NetworkSpec(
                config_path="network.critic_network",
            ),
        },
        optimizer_specs={
            "q_network": OptimizerSpec(
                learning_rate=1e-4,  # Default, overridden by hyperparams
                clip_norm=1.0,
                eps=1e-8,
            ),
        },
        env_action_config=True,
    )


def build_dqn_network(
    net_config: Any,
    instantiate_fn: Callable,
    env_info: EnvironmentInfo,
    exp_config: BaseExperimentConfig,
) -> Any:
    """Builds the DQN Q-network."""
    logger.debug(f"Received net_config type: {type(net_config)}")
    if not hasattr(net_config, "pre_torso") or not hasattr(net_config, "critic_head"):
        raise ValueError(
            "DQN net_config (from critic_network path) must have 'pre_torso' and 'critic_head'."
        )

    q_network_torso = instantiate_fn(net_config.pre_torso)
    q_network_output_head = instantiate_fn(
        net_config.critic_head,
        action_dim=env_info.act_dim,
        epsilon=exp_config.algorithm.hyperparam.training_epsilon.value,
    )
    return FeedForwardActor(torso=q_network_torso, action_head=q_network_output_head)


def build_dqn_optimizer(opt_spec: OptimizerSpec) -> optax.GradientTransformation:
    """Builds the DQN optimizer."""
    return optax.chain(
        optax.inject_hyperparams(optax.clip_by_global_norm)(max_norm=opt_spec.clip_norm),
        optax.inject_hyperparams(optax.adam)(
            learning_rate=opt_spec.learning_rate, eps=opt_spec.eps
        ),
    )


def build_dqn_distributed_layout(
    env,
    nets_and_opts,
    hyperparam_batch_wrappers,
    normalizer_fns,
    key,
    train_strategy,
    config,
):
    """Creates the initial fully distributed DQN learner state."""
    algo_hp_batch = hyperparam_batch_wrappers["algo"]
    cores_for_dist = (config.training.num_devices * config.training.update_batch_size) or 1

    # Convert the generic HyperparamBatch to the specific DQNVectorizedHyperparams NamedTuple.
    hyperparams_per_core_struct = DQNVectorizedHyperparams(
        critic_lr=algo_hp_batch.critic_lr,
        tau=algo_hp_batch.tau,
        gamma=algo_hp_batch.gamma,
        max_grad_norm=algo_hp_batch.max_grad_norm,
        training_epsilon=algo_hp_batch.training_epsilon,
        evaluation_epsilon=algo_hp_batch.evaluation_epsilon,
        max_abs_reward=algo_hp_batch.max_abs_reward,
        huber_loss_parameter=algo_hp_batch.huber_loss_parameter,
        warmup_rollout_length=algo_hp_batch.warmup_rollout_length,
        rollout_length=algo_hp_batch.rollout_length,
        epochs=algo_hp_batch.epochs,
        total_buffer_size=algo_hp_batch.total_buffer_size,
        decay_learning_rates=algo_hp_batch.decay_learning_rates,
        normalize_observations=algo_hp_batch.normalize_observations,
        sample_id=algo_hp_batch.sample_id,
        # Per-Core Calculation Rationale:
        # total_num_envs: The total parallel environments are distributed across devices.
        # total_batch_size: For off-policy updates, each device samples a fraction of the total batch.
        # The gradients are then averaged, resulting in an update step equivalent to the total batch size.
        # E.g., 4 devices with total_batch_size=1024 -> each device samples 256 -> gradients are averaged ->
        # effective update uses 1024 samples.
        total_num_envs=(algo_hp_batch.total_num_envs // cores_for_dist).astype(jnp.int32),
        total_batch_size=(algo_hp_batch.total_batch_size // cores_for_dist).astype(jnp.int32),
    )

    init_norm_fn, _, _ = normalizer_fns
    norm_params_init = init_norm_fn(
        brax_specs.Array((int(config.env.obs_dim),), jnp.dtype("float32"))
    )

    max_dims = get_dqn_max_masked_dims(
        algo_hp_batch, config.training.num_devices, config.training.update_batch_size
    )
    buffer_fns = build_masked_item_buffer(
        max_parallel_envs=max_dims.num_envs,
        buffer_size_per_env=max_dims.buffer_size // max_dims.num_envs,
        effective_buffer_size_per_env=max_dims.buffer_size,
        sample_batch_size=max_dims.batch_size,
    )

    initial_buffer_state = buffer_fns.init(
        DQNTransition(
            obs=jax.tree_util.tree_map(lambda x: x.squeeze(0), nets_and_opts.init_x),
            action=jnp.zeros((), dtype=jnp.int32),
            reward=jnp.zeros((), dtype=jnp.float32),
            done=jnp.zeros((), dtype=bool),
            next_obs=jax.tree_util.tree_map(lambda x: x.squeeze(0), nets_and_opts.init_x),
            info={
                "episode_return": jnp.array(0.0),
                "episode_length": jnp.array(0, dtype=jnp.int32),
                "is_terminal_step": jnp.array(False),
                "valid": jnp.array(0.0),
            },
        )
    )

    initial_components = {
        "params": OnlineAndTarget(
            nets_and_opts.params["q_network"], nets_and_opts.params["q_network"]
        ),
        "opt_states": nets_and_opts.opt_states["q_network"],
        "buffer_state": initial_buffer_state,
        "normalization_params": norm_params_init,
        "total_env_steps_counter": jnp.array(0.0, dtype=jnp.float32),
    }
    hyperparam_structs = {"algo_hyperparams": hyperparams_per_core_struct}

    distributed_state = build_generic_distributed_state(
        DQNLearnerState,
        initial_components,
        hyperparam_structs,
        env,
        key,
        train_strategy,
        max_dims.num_envs,
    )

    eff_buf_size_per_hp = (
        hyperparams_per_core_struct.total_buffer_size
        // hyperparams_per_core_struct.total_num_envs
    ).astype(jnp.int32)
    eff_buf_size_broadcasted = broadcast_hp_batched_array_to_strategy_shape(
        eff_buf_size_per_hp, train_strategy, "hyperparam"
    )

    final_learner_state = distributed_state._replace(
        buffer_state=distributed_state.buffer_state.replace(
            effective_buffer_size_per_env=eff_buf_size_broadcasted
        )
    )
    return final_learner_state

def setup_dqn_keys(base_key: chex.PRNGKey) -> tuple[chex.PRNGKey, ...]:
    """Generates keys needed for DQN: main_learner_key, eval_key, q_network_init_key."""
    key_main, eval_key, q_net_key = jax.random.split(base_key, num=3)
    return key_main, eval_key, q_net_key


def get_dqn_eval_act_fn(
    config: BaseExperimentConfig,
    actor_apply_fn: Callable[[FrozenDict, chex.Array], Any],
) -> ActFn:
    return get_distribution_act_fn(config, actor_apply_fn)


def extract_dqn_online_q_params(
    learner_state: DQNLearnerState,
    train_strategy: DistributionStrategy,
    eval_strategy: DistributionStrategy,
) -> FrozenDict:
    """Extracts online Q-params and slices them for evaluation strategy."""
    online_q_params_train_layout = learner_state.params.online
    return slice_extra_batch_dims(online_q_params_train_layout, train_strategy, eval_strategy)


def extract_common_norm_params_dqn(
    learner_state: DQNLearnerState,
    train_strategy: DistributionStrategy,
    eval_strategy: DistributionStrategy,
) -> NormParams:
    """Extracts normalization params and slices them for evaluation strategy."""
    norm_params_train_layout = learner_state.normalization_params
    return slice_extra_batch_dims(norm_params_train_layout, train_strategy, eval_strategy)


def build_dqn_algo_setup_fns_for_phase_training() -> AlgoSetupFns:
    """Factory that creates the AlgoSetupFns tuple for DQN, for the Phaser."""

    return build_generic_phaser_setup_fns(
        build_network_setup_fn=build_dqn_network_setup,
        build_network_fn=build_dqn_network,
        build_optimizer_fn=build_dqn_optimizer,
        build_update_step_fn=build_dqn_update_step_fn,
        build_distributed_layout_fn=build_dqn_distributed_layout,
        build_warmup_rollout_fn=build_warmup_rollout_fn,
        get_eval_act_fn_callback=get_dqn_eval_act_fn,
        extract_params_for_eval_fn=extract_dqn_online_q_params,
        extract_norm_params_for_eval_fn=extract_common_norm_params_dqn,
    )
