import dataclasses
import logging
from collections.abc import Callable
from typing import Any

import brax.training.acme.specs as brax_specs
import chex
import jax
import jax.numpy as jnp
import optax
from flax.core.frozen_dict import FrozenDict

from hyperlax.algo.sac.core import build_sac_update_step_fn, build_warmup_rollout_fn
from hyperlax.algo.sac.struct_sac import (
    SACLearnerState,
    SACOptStates,
    SACParams,
    SACTransition,
    SACVectorizedHyperparams,
    get_sac_max_masked_dims,
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
from hyperlax.network.base import CompositeArchitecture, FeedForwardActor, MultiNetwork
from hyperlax.normalizer.running_stats import NormParams
from hyperlax.rbuffer.masked_item_buffer import build_masked_item_buffer
from hyperlax.trainer.phaser import AlgoSetupFns, build_generic_phaser_setup_fns
from hyperlax.utils.algo_setup import (
    AlgorithmNetworkSetup,
    EnvironmentInfo,
    NetworkSpec,
    OptimizerSpec,
    StandaloneParameterSpec,
)

logger = logging.getLogger(__name__)


def build_sac_network_setup() -> AlgorithmNetworkSetup:
    """Create SAC-specific network setup."""
    return AlgorithmNetworkSetup(
        network_specs={
            "actor_network": NetworkSpec(config_path="network.actor_network"),
            "critic_network": NetworkSpec(config_path="network.critic_network"),
        },
        param_specs={
            "alpha": StandaloneParameterSpec(init_fn=lambda: jnp.array(0.0, dtype=jnp.float32))
        },
        optimizer_specs={
            "actor_network": OptimizerSpec(learning_rate=3e-4, clip_norm=1.0),
            "critic_network": OptimizerSpec(learning_rate=3e-4, clip_norm=1.0),
            "alpha": OptimizerSpec(learning_rate=3e-4, clip_norm=1.0),
        },
        env_action_config=True,
    )


def build_sac_network(
    net_config: Any,
    instantiate_fn: Callable,
    env_info: EnvironmentInfo,
    exp_config: BaseExperimentConfig,
) -> Any:
    """Builds SAC actor or critic networks."""
    if hasattr(net_config, "action_head"):  # Actor
        if env_info.action_space_type != "continuous":
            raise TypeError(
                f"SAC with NormalAffineTanhDistributionHead is incompatible with "
                f"discrete action spaces. Environment '{exp_config.env.scenario.name}' "
                f"has action space type '{env_info.action_space_type}'."
            )
        torso = instantiate_fn(net_config.pre_torso)
        action_head = instantiate_fn(
            net_config.action_head,
            action_dim=env_info.act_dim,
            minimum=env_info.act_minimum,
            maximum=env_info.act_maximum,
        )
        return FeedForwardActor(torso=torso, action_head=action_head)
    elif hasattr(net_config, "critic_head"):  # Critic

        def create_single_q_network() -> CompositeArchitecture:
            """A factory function to create one instance of the Q-network."""
            input_layer = instantiate_fn(net_config.input_layer)
            torso_input_dim = env_info.obs_dim
            if "ObservationActionInput" in getattr(net_config.input_layer, "_target_", ""):
                torso_input_dim += env_info.act_dim

            modified_torso_config = net_config.pre_torso
            if hasattr(modified_torso_config, "in_dim"):
                modified_torso_config = dataclasses.replace(
                    modified_torso_config, in_dim=torso_input_dim
                )
            elif hasattr(modified_torso_config, "n_features"):
                modified_torso_config = dataclasses.replace(
                    modified_torso_config, n_features=torso_input_dim
                )

            torso = instantiate_fn(modified_torso_config)
            head = instantiate_fn(net_config.critic_head)
            return CompositeArchitecture([input_layer, torso, head])

        # SAC requires two Q-networks for the critic.
        return MultiNetwork([create_single_q_network(), create_single_q_network()])
    raise ValueError(f"Unknown SAC network config: {net_config}")


def build_sac_optimizer(opt_spec: OptimizerSpec) -> optax.GradientTransformation:
    """Builds an Adam optimizer for SAC components."""
    return optax.chain(
        optax.inject_hyperparams(optax.clip_by_global_norm)(max_norm=opt_spec.clip_norm),
        optax.inject_hyperparams(optax.adam)(
            learning_rate=opt_spec.learning_rate, eps=opt_spec.eps
        ),
    )


def build_sac_distributed_layout(
    env,
    nets_and_opts,
    hyperparam_batch_wrappers,
    normalizer_fns,
    key,
    train_strategy,
    config,
):

    # 1. Get the generic HP batch and calculate per-core values.
    algo_hp_batch = hyperparam_batch_wrappers["algo"]
    cores_for_dist = (config.training.num_devices * config.training.update_batch_size) or 1

    # 2. Convert the generic HyperparamBatch into the specific SACVectorizedHyperparams NamedTuple.
    hyperparams_per_core_struct = SACVectorizedHyperparams(
        actor_lr=algo_hp_batch.actor_lr,
        q_lr=algo_hp_batch.q_lr,
        alpha_lr=algo_hp_batch.alpha_lr,
        gamma=algo_hp_batch.gamma,
        tau=algo_hp_batch.tau,
        max_grad_norm=algo_hp_batch.max_grad_norm,
        target_entropy_scale=algo_hp_batch.target_entropy_scale,
        init_alpha=algo_hp_batch.init_alpha,
        rollout_length=algo_hp_batch.rollout_length,
        warmup_rollout_length=algo_hp_batch.warmup_rollout_length,
        epochs=algo_hp_batch.epochs,
        total_buffer_size=algo_hp_batch.total_buffer_size,
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

    max_dims = get_sac_max_masked_dims(
        algo_hp_batch, config.training.num_devices, config.training.update_batch_size
    )

    # 3. Instantiate normalization parameters.
    init_norm_fn, _, _ = normalizer_fns
    initial_norm_params = init_norm_fn(
        brax_specs.Array((int(config.env.obs_dim),), jnp.dtype("float32"))
    )

    # 4. Instantiate buffer.
    buffer_fns = build_masked_item_buffer(
        max_parallel_envs=max_dims.num_envs,
        buffer_size_per_env=max_dims.buffer_size // max_dims.num_envs,
        effective_buffer_size_per_env=max_dims.buffer_size,
        sample_batch_size=max_dims.batch_size,
    )

    initial_buffer_state = buffer_fns.init(
        SACTransition(
            obs=jax.tree_util.tree_map(lambda x: x.squeeze(0), nets_and_opts.init_x),
            action=jnp.zeros(env.action_spec().shape, dtype=jnp.float32),
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

    # 5. Group raw components (single-instance parameters, opt_states, etc.) for the generic builder
    initial_params_raw = SACParams(
        actor_params=nets_and_opts.params["actor_network"],
        q_params=OnlineAndTarget(
            online=nets_and_opts.params["critic_network"],
            target=nets_and_opts.params["critic_network"],
        ),
        log_alpha=nets_and_opts.params["alpha"],
    )

    initial_opt_states_raw = SACOptStates(
        actor_opt_state=nets_and_opts.opt_states["actor_network"],
        q_opt_state=nets_and_opts.opt_states["critic_network"],
        alpha_opt_state=nets_and_opts.opt_states["alpha"],
    )

    initial_components_for_generic_builder = {
        "params": initial_params_raw,
        "opt_states": initial_opt_states_raw,
        "buffer_state": initial_buffer_state,
        "normalization_params": initial_norm_params,
        "total_env_steps_counter": jnp.array(0.0, dtype=jnp.float32),
    }

    # 6. Call the generic builder with the specific NamedTuple struct
    final_learner_state = build_generic_distributed_state(
        SACLearnerState,
        initial_components_for_generic_builder,
        {"algo_hyperparams": hyperparams_per_core_struct},
        env,
        key,
        train_strategy,
        max_dims.num_envs,
    )

    # 7. Post-distribution adjustments
    distributed_hyperparams_struct = final_learner_state.algo_hyperparams
    initial_log_alpha_dist_from_hp = jnp.log(distributed_hyperparams_struct.init_alpha)
    final_learner_state = final_learner_state._replace(
        params=final_learner_state.params._replace(log_alpha=initial_log_alpha_dist_from_hp)
    )

    # Correct effective_buffer_size_per_env based on distributed hyperparams
    # Previously only have buffer size derived from max dims,
    # now setting individual effective buffer size to be used by masking logic in maskeditembuffer
    eff_buf_size_per_hp = (
        hyperparams_per_core_struct.total_buffer_size
        // hyperparams_per_core_struct.total_num_envs
    ).astype(jnp.int32)
    eff_buf_size_per_hp_dist = broadcast_hp_batched_array_to_strategy_shape(
        eff_buf_size_per_hp, train_strategy, "hyperparam"
    )
    final_learner_state = final_learner_state._replace(
        buffer_state=final_learner_state.buffer_state.replace(
            effective_buffer_size_per_env=eff_buf_size_per_hp_dist
        )
    )
    return final_learner_state


def setup_sac_keys(base_key: chex.PRNGKey) -> tuple[chex.PRNGKey, ...]:
    return jax.random.split(base_key, 5)  # main, eval, actor, critic, log_alpha


def get_sac_eval_act_fn(
    config: BaseExperimentConfig,
    actor_apply_fn: Callable[[FrozenDict, chex.Array], Any],
) -> ActFn:
    return get_distribution_act_fn(config, actor_apply_fn)


def extract_sac_actor_params(
    learner_state: SACLearnerState,
    train_strategy: DistributionStrategy,
    eval_strategy: DistributionStrategy,
) -> FrozenDict:
    actor_params_train_layout = learner_state.params.actor_params
    return slice_extra_batch_dims(actor_params_train_layout, train_strategy, eval_strategy)


def extract_common_norm_params_sac(
    learner_state: SACLearnerState,
    train_strategy: DistributionStrategy,
    eval_strategy: DistributionStrategy,
) -> NormParams:
    norm_params_train_layout = learner_state.normalization_params
    return slice_extra_batch_dims(norm_params_train_layout, train_strategy, eval_strategy)


def build_sac_algo_setup_fns_for_phase_training() -> AlgoSetupFns:
    """Factory that creates the AlgoSetupFns tuple for SAC."""
    return build_generic_phaser_setup_fns(
        build_network_setup_fn=build_sac_network_setup,
        build_network_fn=build_sac_network,
        build_optimizer_fn=build_sac_optimizer,
        build_update_step_fn=build_sac_update_step_fn,
        build_distributed_layout_fn=build_sac_distributed_layout,
        build_warmup_rollout_fn=build_warmup_rollout_fn,
        get_eval_act_fn_callback=get_sac_eval_act_fn,
        extract_params_for_eval_fn=extract_sac_actor_params,
        extract_norm_params_for_eval_fn=extract_common_norm_params_sac,
    )
