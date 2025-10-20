import logging
from collections.abc import Callable
from typing import Any

import brax.training.acme.specs as brax_specs
import chex
import jax
import jax.numpy as jnp
import optax
from flax.core.frozen_dict import FrozenDict

from hyperlax.algo.ppo.core import build_ppo_update_step_fn
from hyperlax.algo.ppo.struct_ppo import (
    PPOOnPolicyLearnerState,
    PPOVectorizedHyperparams,
    get_ppo_max_masked_dims,
)
from hyperlax.base_types import ActFn, ActorCriticOptStates, ActorCriticParams
from hyperlax.configs.main_base import BaseExperimentConfig
from hyperlax.evaluator.core import get_distribution_act_fn
from hyperlax.evaluator.setup import slice_extra_batch_dims
from hyperlax.layout.axes import DistributionStrategy
from hyperlax.layout.data import build_generic_distributed_state
from hyperlax.network.base import FeedForwardActor, FeedForwardCritic
from hyperlax.network.hyperparam import MLPVectorizedHyperparams
from hyperlax.normalizer.running_stats import NormParams
from hyperlax.trainer.phaser import AlgoSetupFns, build_generic_phaser_setup_fns
from hyperlax.utils.algo_setup import (
    AlgorithmNetworkSetup,
    EnvironmentInfo,
    NetworkSpec,
    OptimizerSpec,
)

logger = logging.getLogger(__name__)


def build_ppo_network_setup() -> AlgorithmNetworkSetup:
    """Creates the network and optimizer specification for PPO."""
    return AlgorithmNetworkSetup(
        network_specs={
            "actor_network": NetworkSpec(config_path="network.actor_network"),
            "critic_network": NetworkSpec(config_path="network.critic_network"),
        },
        optimizer_specs={
            "actor_network": OptimizerSpec(learning_rate=3e-4, clip_norm=1.0),
            "critic_network": OptimizerSpec(learning_rate=3e-4, clip_norm=1.0),
        },
        env_action_config=True,
    )

def build_ppo_network(
    net_config: Any,
    instantiate_fn: Callable,
    env_info: EnvironmentInfo,
    exp_config: BaseExperimentConfig,
) -> Any:
    logger.debug(f"Received net_config type: {type(net_config)}, content: {net_config}")
    logger.debug(f"env_info: {env_info}")

    if hasattr(net_config, "action_head"):
        actor_torso_config = net_config.pre_torso
        actor_torso = instantiate_fn(actor_torso_config)

        action_head_config_obj = net_config.action_head
        head_target_path = getattr(action_head_config_obj, "_target_", "")

        head_init_kwargs = {"action_dim": env_info.act_dim}

        is_categorical_head = "CategoricalHead" in head_target_path
        is_continuous_head = (
            "NormalAffineTanhDistributionHead" in head_target_path
            or "BetaDistributionHead" in head_target_path
        )

        if env_info.action_space_type == "discrete":
            if not is_categorical_head:
                logger.error(
                    f"FATAL MISMATCH: Environment is DISCRETE, "
                    f"but PPO Actor head is configured as '{head_target_path}' (continuous). "
                    f"Please use a recipe with a discrete action head (e.g., one targeting CategoricalHead)."
                )
                raise ValueError(
                    f"PPO Actor head type '{head_target_path}' incompatible with discrete environment."
                )
            logger.debug(
                f"Instantiating discrete head ({head_target_path}) with action_dim: {env_info.act_dim}"
            )
        elif env_info.action_space_type == "continuous":
            if not is_continuous_head:
                logger.error(
                    f"FATAL MISMATCH: Environment is CONTINUOUS, "
                    f"but PPO Actor head is configured as '{head_target_path}' (discrete). "
                    f"Please use a recipe with a continuous action head."
                )
                raise ValueError(
                    f"PPO Actor head type '{head_target_path}' incompatible with continuous environment."
                )

            min_val_for_head = getattr(action_head_config_obj, "minimum", None)
            max_val_for_head = getattr(action_head_config_obj, "maximum", None)

            if min_val_for_head is None or max_val_for_head is None:
                logger.error(
                    f"CRITICAL for CONTINUOUS ENV + CONTINUOUS HEAD: "
                    f"Head {head_target_path} requires min/max, but they are None "
                    f"on its config object. Min: {min_val_for_head}, Max: {max_val_for_head}. "
                    f"These should be set by set_env_dims based on env_info."
                )
                raise ValueError(
                    "Min/Max not set on continuous head config object for continuous environment."
                )

            head_init_kwargs["minimum"] = min_val_for_head
            head_init_kwargs["maximum"] = max_val_for_head
            logger.debug(
                f"Instantiating continuous head ({head_target_path}). Args: {head_init_kwargs}"
            )
        else:
            # This case should ideally not be reached if env_info.action_space_type is always discrete/continuous
            logger.error(f"Unknown environment action space type: {env_info.action_space_type}")
            raise ValueError(
                f"Unsupported environment action space type: {env_info.action_space_type}"
            )

        actor_action_head = instantiate_fn(action_head_config_obj, **head_init_kwargs)
        return FeedForwardActor(torso=actor_torso, action_head=actor_action_head)

    elif hasattr(net_config, "critic_head"):
        critic_torso_config = net_config.pre_torso
        critic_torso = instantiate_fn(critic_torso_config)
        critic_head_config_obj = net_config.critic_head
        critic_head = instantiate_fn(critic_head_config_obj)
        return FeedForwardCritic(torso=critic_torso, critic_head=critic_head)
    else:
        raise ValueError(
            f"Unknown network config structure for PPO. Expected 'action_head' or 'critic_head'. Got: {net_config}"
        )


def build_ppo_optimizer(opt_spec: OptimizerSpec) -> optax.GradientTransformation:
    """Builds the PPO optimizer."""
    return optax.chain(
        optax.inject_hyperparams(optax.clip_by_global_norm)(max_norm=opt_spec.clip_norm),
        optax.inject_hyperparams(optax.adam)(
            learning_rate=opt_spec.learning_rate, eps=opt_spec.eps
        ),
    )


def build_ppo_distributed_layout(
    env,
    nets_and_opts,
    hyperparam_batch_wrappers,
    normalizer_fns,
    key,
    train_strategy,
    config,
):
    """Creates the initial fully distributed PPO learner state."""
    # 1. Get the generic HP batch and calculate per-core values.
    algo_hp_batch = hyperparam_batch_wrappers["algo"]
    actor_network_hp_batch = hyperparam_batch_wrappers.get("network_actor")
    critic_network_hp_batch = hyperparam_batch_wrappers.get("network_critic")
    cores_for_dist = (config.training.num_devices * config.training.update_batch_size) or 1

    # 2. Convert generic HyperparamBatch objects to specific NamedTuples.
    ppo_hyperparams_per_core_struct = PPOVectorizedHyperparams(
        actor_lr=algo_hp_batch.actor_lr,
        critic_lr=algo_hp_batch.critic_lr,
        gamma=algo_hp_batch.gamma,
        gae_lambda=algo_hp_batch.gae_lambda,
        clip_eps=algo_hp_batch.clip_eps,
        ent_coef=algo_hp_batch.ent_coef,
        vf_coef=algo_hp_batch.vf_coef,
        max_grad_norm=algo_hp_batch.max_grad_norm,
        rollout_length=algo_hp_batch.rollout_length,
        epochs=algo_hp_batch.epochs,
        num_minibatches=algo_hp_batch.num_minibatches,
        standardize_advantages=algo_hp_batch.standardize_advantages,
        decay_learning_rates=algo_hp_batch.decay_learning_rates,
        normalize_observations=algo_hp_batch.normalize_observations,
        sample_id=algo_hp_batch.sample_id,
        # Per-Core Calculation Rationale:
        # total_num_envs: The total parallel environments are distributed across devices.
        # total_batch_size (Implicit): PPO's "batch size" is the rollout data, which is a function
        # of rollout_length * total_num_envs (per core). num_minibatches then determines how this
        # per-core data is split for updates. It's an on-policy algorithm, so unlike SAC/DQN,
        # there is no shared replay buffer from which a global total_batch_size is sampled.
        total_num_envs=(algo_hp_batch.total_num_envs // cores_for_dist).astype(jnp.int32),
    )

    actor_net_hps_struct = (
        MLPVectorizedHyperparams(
            *[getattr(actor_network_hp_batch, field) for field in MLPVectorizedHyperparams._fields]
        )
        if actor_network_hp_batch
        else None
    )
    critic_net_hps_struct = (
        MLPVectorizedHyperparams(
            *[
                getattr(critic_network_hp_batch, field)
                for field in MLPVectorizedHyperparams._fields
            ]
        )
        if critic_network_hp_batch
        else None
    )

    # 3. Get max dimensions for padding.
    max_dims = get_ppo_max_masked_dims(
        algo_hp_batch, config.training.num_devices, config.training.update_batch_size
    )

    # 4. Initialize non-HP components.
    init_norm_fn, _, _ = normalizer_fns
    initial_norm_params = init_norm_fn(
        brax_specs.Array((int(config.env.obs_dim),), jnp.dtype("float32"))
    )

    initial_components_for_builder = {
        "params": ActorCriticParams(
            actor_params=nets_and_opts.params["actor_network"],
            critic_params=nets_and_opts.params["critic_network"],
        ),
        "opt_states": ActorCriticOptStates(
            actor_opt_state=nets_and_opts.opt_states["actor_network"],
            critic_opt_state=nets_and_opts.opt_states["critic_network"],
        ),
        "normalization_params": initial_norm_params,
        "total_env_steps_counter": jnp.array(0.0, dtype=jnp.float32),
    }

    hyperparam_structs_for_builder = {
        "algo_hyperparams": ppo_hyperparams_per_core_struct,
        "actor_network_hyperparams": actor_net_hps_struct,
        "critic_network_hyperparams": critic_net_hps_struct,
    }

    # 5. Call the generic builder.
    final_learner_state = build_generic_distributed_state(
        PPOOnPolicyLearnerState,
        initial_components_for_builder,
        hyperparam_structs_for_builder,
        env,
        key,
        train_strategy,
        max_dims.envs,
    )
    return final_learner_state

def setup_ppo_keys(base_key: chex.PRNGKey) -> tuple[chex.PRNGKey, ...]:
    """Generates keys needed for PPO: main_learner_key, eval_key, actor_init_key, critic_init_key."""
    key_main, eval_key, actor_net_key, critic_net_key = jax.random.split(base_key, num=4)
    return key_main, eval_key, actor_net_key, critic_net_key


def get_ppo_eval_act_fn(
    config: BaseExperimentConfig,
    actor_apply_fn: Callable[[FrozenDict, chex.Array], Any],
) -> ActFn:
    return get_distribution_act_fn(config, actor_apply_fn)


def extract_ppo_actor_params(
    learner_state: PPOOnPolicyLearnerState,
    train_strategy: DistributionStrategy,
    eval_strategy: DistributionStrategy,
) -> FrozenDict:
    actor_params_train_layout = learner_state.params.actor_params
    return slice_extra_batch_dims(actor_params_train_layout, train_strategy, eval_strategy)


def extract_common_norm_params_ppo(
    learner_state: PPOOnPolicyLearnerState,
    train_strategy: DistributionStrategy,
    eval_strategy: DistributionStrategy,
) -> NormParams:
    norm_params_train_layout = learner_state.normalization_params
    return slice_extra_batch_dims(norm_params_train_layout, train_strategy, eval_strategy)


def build_ppo_algo_setup_fns_for_phase_training() -> AlgoSetupFns:
    """Factory that creates the AlgoSetupFns tuple for PPO, for the Phaser."""

    return build_generic_phaser_setup_fns(
        build_network_setup_fn=build_ppo_network_setup,
        build_network_fn=build_ppo_network,
        build_optimizer_fn=build_ppo_optimizer,
        build_update_step_fn=build_ppo_update_step_fn,
        build_distributed_layout_fn=build_ppo_distributed_layout,
        build_warmup_rollout_fn=None,  # PPO is on-policy, no warmup
        get_eval_act_fn_callback=get_ppo_eval_act_fn,
        extract_params_for_eval_fn=extract_ppo_actor_params,
        extract_norm_params_for_eval_fn=extract_common_norm_params_ppo,
    )
