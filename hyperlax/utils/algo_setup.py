"""
Algorithm-agnostic setup utilities for RL training.

This module provides functional, composable utilities for setting up networks,
optimizers, and distributed state for any RL algorithm.
"""

import dataclasses
import logging
from collections.abc import Callable
from dataclasses import (
    dataclass,
)
from typing import (
    TYPE_CHECKING,
    Any,
    Literal,
    NamedTuple,
    TypeVar,
)

import jax
import jax.numpy as jnp
from jumanji.env import Environment

from hyperlax.algo.ppo.config import PPOConfig
from hyperlax.algo.sac.config import SACConfig
from hyperlax.base_types import AnakinTrainOutput
from hyperlax.configs.main_base import BaseExperimentConfig
from hyperlax.env.utils import identify_action_space_type
from hyperlax.layout.axes import DistributionStrategy
from hyperlax.layout.ops import transform_function_by_strategy
from hyperlax.logger.hp_progress_bar import ProgressDisplayObserver, display_hp_progress
from hyperlax.network.utils import instantiate_from_config
from hyperlax.trainer.utils import _calculate_milestones

if TYPE_CHECKING:
    from hyperlax.base_types import AlgorithmGlobalSetupArgs

LearnerFnT = TypeVar("LearnerFnT")
LearnerStateT = TypeVar("LearnerStateT")
logger = logging.getLogger(__name__)


class EnvironmentInfo(NamedTuple):  # TODO rename to specific after migration
    """Container for environment-extracted information."""

    act_dim: int
    act_minimum: float
    act_maximum: float
    action_space_type: str
    obs_dim: int | None = None
    obs_spec: Any = None


class NetworkBuildComponents(NamedTuple):
    """Result of single network building."""

    network: Any
    params: Any | None
    opt_state: Any | None = None


class NetworkAndOptimizerOnlyFns(NamedTuple):
    """Result of function-only building."""

    networks: dict[str, Any]
    optimizers: dict[str, Any]
    env_info: EnvironmentInfo


class NetworkAndOptimizerWithParamsTuple(NamedTuple):
    """Result of full networks and optimizers building."""

    networks: dict[str, Any]
    params: dict[str, Any]
    opt_states: dict[str, Any]
    optimizers: dict[str, Any]
    env_info: EnvironmentInfo
    init_x: Any | None = None


@dataclass(frozen=True)
class NetworkSpec:
    """Specification for building a network component."""

    config_path: str  # Path to config (e.g., 'network.actor_network')
    init_args: dict[str, Any] | None = None


@dataclass(frozen=True)
class StandaloneParameterSpec:
    """Specification for a standalone, non-network parameter (e.g., SAC's alpha)."""

    init_fn: Callable[..., Any]  # A function to create the initial parameter, e.g., jnp.zeros


@dataclass(frozen=True)
class OptimizerSpec:
    """Specification for an optimizer configuration."""

    learning_rate: float = 1e-3
    clip_norm: float = 1.0
    eps: float = 1e-8
    beta1: float = 0.9
    beta2: float = 0.999
    weight_decay: float = 0.0


@dataclass(frozen=True)
class AlgorithmNetworkSetup:
    """Complete specification for algorithm network setup."""

    network_specs: dict[str, NetworkSpec]  # component_id -> NetworkSpec
    optimizer_specs: dict[str, OptimizerSpec]  # component_id -> OptimizerSpec
    param_specs: dict[str, StandaloneParameterSpec] = dataclasses.field(default_factory=dict)
    env_action_config: bool = True


def extract_environment_info(env) -> EnvironmentInfo:
    obs_spec = env.observation_spec()
    act_spec = env.action_spec()
    action_space_type = identify_action_space_type(env)
    obs_dim = obs_spec.agent_view.shape[0]
    if action_space_type == "discrete":
        act_dim = int(act_spec.num_values)
        act_minimum = None
        act_maximum = None
    elif action_space_type == "continuous":
        act_dim = int(act_spec.shape[0])
        act_minimum = float(jnp.min(act_spec.minimum))
        act_maximum = float(jnp.max(act_spec.maximum))
    else:
        raise ValueError(f"Unsupported action space type: {action_space_type}")
    return EnvironmentInfo(
        act_dim=act_dim,
        act_minimum=act_minimum,
        act_maximum=act_maximum,
        obs_dim=obs_dim,
        obs_spec=obs_spec,
        action_space_type=action_space_type,
    )


def get_config_by_path(config, path: str) -> Any:
    """Get configuration object by dot-separated path."""
    parts = path.split(".")
    result = config
    for part in parts:
        result = getattr(result, part)
    return result


def get_network_init_input(
    env: Any, net_name: str, env_info: EnvironmentInfo, algo_context_str: str
) -> tuple:
    """
    Get appropriate initialization input(s) for a network, returned as a tuple.
    This allows for networks that require multiple inputs for their init call.
    """
    """Get appropriate initialization input for network."""
    init_x = env.observation_spec().generate_value()
    init_obs_batch = jax.tree_util.tree_map(lambda x: x[None, ...], init_x)
    # Convention: if 'critic' is in the network name and it's a continuous
    # action space algorithm (like SAC), assume it's a Q-network that needs
    # both observation and action for initialization.
    if (
        "critic" in net_name.lower()
        and "sac" in algo_context_str
        and env_info.action_space_type == "continuous"
    ):
        dummy_action = jnp.zeros((1, env_info.act_dim))
        return (init_obs_batch, dummy_action)
    return (init_obs_batch,)


def build_networks_and_optimizers(
    network_setup: AlgorithmNetworkSetup,
    config: Any,
    env: Any,
    network_keys: dict[str, jax.random.PRNGKey],
    hyperparam_batch: Any,
    instantiate_fn: Callable,
    network_builder_fn: Callable,
    optimizer_builder_fn: Callable,
    mode: Literal[
        "give_me_only_fns", "give_me_fns_and_init_params"
    ] = "give_me_fns_and_init_params",
) -> NetworkAndOptimizerOnlyFns | NetworkAndOptimizerWithParamsTuple:
    """
    Build networks and optimizers for an algorithm.

    Args:
        network_setup: Algorithm network setup specification
        config: Configuration object
        env: Environment instance
        network_keys: Dictionary of network names to JAX keys (ignored in give_me_only_fns mode)
        hyperparam_batch: Hyperparameter batch for dimension extraction
        instantiate_fn: Function to instantiate from config
        network_builder_fn: Algorithm-specific network builder function
        optimizer_builder_fn: Algorithm-specific optimizer builder function
        mode: "give_me_only_fns" or "give_me_fns_and_init_params"

    Returns:
        NetworkOnlyComponents if mode="give_me_only_fns"
        NetworkAndOptimizerTuple if mode="give_me_fns_and_init_params"
    """
    # Extract environment information
    env_info = extract_environment_info(env)
    if network_setup.env_action_config:
        config = update_config_with_env_info(config, env)  # Use the new immutable set_env_dims

    # The check for 'get_max_masked_dims' has been removed as it was incorrect.
    # The new generic HyperparamBatch does not have this method.

    networks = {}
    optimizers = {}

    config_root_for_paths = config.algorithm

    # Build all specified optimizers first, as they can be for networks or standalone params
    for component_id, opt_spec in network_setup.optimizer_specs.items():
        optimizers[component_id] = optimizer_builder_fn(opt_spec)

    # Build all networks
    algo_context_str = network_builder_fn.__name__.lower()
    for net_name, net_spec in network_setup.network_specs.items():
        # Get the full network component config (e.g., MLPActorNetworkConfig)
        net_config_component = get_config_by_path(config_root_for_paths, net_spec.config_path)

        # The builder now receives the full component config.
        network = network_builder_fn(net_config_component, instantiate_fn, env_info, config)
        networks[net_name] = network

    # Return early if only functions are requested
    if mode == "give_me_only_fns":
        return NetworkAndOptimizerOnlyFns(
            networks=networks,
            optimizers=optimizers,
            env_info=env_info,
        )

    # Full mode: initialize parameters and optimizer states
    params = {}
    opt_states = {}

    # Create a single representative dummy observation needed by off-policy algos
    # We just need the structure, so we can generate it once.
    init_x = env.observation_spec().generate_value()
    init_x = jax.tree_util.tree_map(lambda x: x[None, ...], init_x)

    for net_name, network in networks.items():
        if net_name not in network_keys:
            continue

        # Initialize parameters
        init_args = get_network_init_input(env, net_name, env_info, algo_context_str)
        # Unpack the tuple to pass arguments to init: network.init(rng, obs, action)
        network_params = network.init(network_keys[net_name], *init_args)

        # Store results
        params[net_name] = network_params

        # Initialize optimizer state if optimizer exists
        if net_name in optimizers:
            opt_states[net_name] = optimizers[net_name].init(network_params)

    # Initialize standalone parameters and their optimizer states
    for param_name, param_spec in network_setup.param_specs.items():
        if param_name not in network_keys:
            continue
        # Initialize the parameter using its init_fn
        standalone_param = param_spec.init_fn()
        params[param_name] = standalone_param
        if param_name in optimizers:
            opt_states[param_name] = optimizers[param_name].init(standalone_param)

    return NetworkAndOptimizerWithParamsTuple(
        networks=networks,
        params=params,
        opt_states=opt_states,
        optimizers=optimizers,
        env_info=env_info,
        init_x=init_x,
    )


def update_config_with_env_info(
    config: BaseExperimentConfig, env: Environment
) -> BaseExperimentConfig:
    """
    Updates the configuration object with dimensions extracted from the environment.
    This is an immutable operation that returns a new, updated config object.
    It consolidates all environment-dependent configuration patching:
    1. Updates `config.env` with `obs_dim`, `act_dim`, etc.
    2. Explicitly patches network torso input dimensions (e.g., `n_features`, `in_dim`).
    3. Explicitly patches algorithm-specific network heads (e.g., PPO action head).
    """
    env_info: EnvironmentInfo = extract_environment_info(env)

    # 1. Update config.env with dimensions
    new_env_config = dataclasses.replace(
        config.env,
        obs_dim=env_info.obs_dim,
        act_dim=env_info.act_dim,
        act_minimum=env_info.act_minimum,
        act_maximum=env_info.act_maximum,
    )
    config = dataclasses.replace(config, env=new_env_config)
    logger.debug(f"Updating config with env_info: {env_info}")

    # 2. Explicitly patch network configs
    if not hasattr(config.algorithm, "network"):
        return config

    algo_config = config.algorithm
    network_config = algo_config.network
    network_config_changes = {}

    def _patch_torso(torso_cfg: Any, input_dim: int) -> Any:
        """Helper to immutably patch torso input dimensions."""
        replacements = {}
        if hasattr(torso_cfg, "n_features") and getattr(torso_cfg, "n_features", 0) <= 0:
            replacements["n_features"] = input_dim
        if hasattr(torso_cfg, "in_dim") and getattr(torso_cfg, "in_dim", 1) <= 0:
            replacements["in_dim"] = input_dim
        if hasattr(torso_cfg, "input_dim") and getattr(torso_cfg, "input_dim", 0) <= 0:
            replacements["input_dim"] = input_dim

        if replacements:
            logger.info(
                f"Patching torso '{type(torso_cfg).__name__}' with input dimensions: {replacements}"
            )
            return dataclasses.replace(torso_cfg, **replacements)
        return torso_cfg

    # --- Actor Network Patching ---
    if hasattr(network_config, "actor_network"):
        actor_net_cfg = network_config.actor_network
        actor_net_changes = {}

        # Patch torso with observation dimension
        new_actor_torso = _patch_torso(actor_net_cfg.pre_torso, env_info.obs_dim)
        if new_actor_torso is not actor_net_cfg.pre_torso:
            actor_net_changes["pre_torso"] = new_actor_torso

        # Patch PPO/SAC action heads
        if isinstance(algo_config, PPOConfig):
            new_action_head_cfg = None
            if env_info.action_space_type == "discrete":

                @dataclasses.dataclass
                class CategoricalActionHeadConfig:
                    _target_: str = "hyperlax.network.heads.CategoricalHead"

                new_action_head_cfg = CategoricalActionHeadConfig()
                logger.info("PPO recipe detected discrete env. Setting action head to CategoricalHead.")
            elif env_info.action_space_type == "continuous":

                @dataclasses.dataclass(frozen=True)
                class NormalAffineTanhDistributionHeadConfig:
                    _target_: str = "hyperlax.network.heads.NormalAffineTanhDistributionHead"
                    minimum: float = -999.0
                    maximum: float = -999.0

                new_action_head_cfg = NormalAffineTanhDistributionHeadConfig(
                    minimum=float(env_info.act_minimum), maximum=float(env_info.act_maximum)
                )
                logger.info(
                    "PPO recipe detected continuous env. Setting action head to NormalAffineTanhDistributionHead."
                )

            if new_action_head_cfg:
                actor_net_changes["action_head"] = new_action_head_cfg

        elif isinstance(algo_config, SACConfig):
            # SAC head needs min/max values from env
            new_head = dataclasses.replace(
                actor_net_cfg.action_head,
                minimum=float(env_info.act_minimum),
                maximum=float(env_info.act_maximum),
            )
            actor_net_changes["action_head"] = new_head

        if actor_net_changes:
            network_config_changes["actor_network"] = dataclasses.replace(
                actor_net_cfg, **actor_net_changes
            )

    # --- Critic/Q Network Patching ---
    if hasattr(network_config, "critic_network"):
        critic_net_cfg = network_config.critic_network

        critic_input_dim = env_info.obs_dim
        # SAC critic is special, it takes obs+action
        if isinstance(algo_config, SACConfig) and hasattr(critic_net_cfg, "input_layer"):
            if "ObservationActionInput" in critic_net_cfg.input_layer._target_:
                critic_input_dim += env_info.act_dim

        new_critic_torso = _patch_torso(critic_net_cfg.pre_torso, critic_input_dim)
        if new_critic_torso is not critic_net_cfg.pre_torso:
            # For DQN, critic_network is the Q-network.
            network_config_changes["critic_network"] = dataclasses.replace(
                critic_net_cfg, pre_torso=new_critic_torso
            )

    # Apply all accumulated changes
    if network_config_changes:
        new_network_config = dataclasses.replace(network_config, **network_config_changes)
        new_algo_config = dataclasses.replace(algo_config, network=new_network_config)
        config = dataclasses.replace(config, algorithm=new_algo_config)

    return config


def setup_generic_learner(
    global_args: "AlgorithmGlobalSetupArgs",
    hyperparam_batch_wrappers: dict[str, Any],
    hyperparam_non_vectorizeds: Any,
    num_updates_per_scan: int,
    train_strategy: DistributionStrategy,
    # --- Algorithm-specific Callbacks ---
    build_network_setup_fn: Callable[[], AlgorithmNetworkSetup],
    build_network_fn: Callable,
    build_optimizer_fn: Callable,
    build_update_step_fn: Callable,
    build_distributed_layout_fn: Callable,
    with_params: bool,
    build_warmup_rollout_fn: Callable | None = None,
) -> tuple[
    Callable[[LearnerStateT], AnakinTrainOutput],
    Callable[[LearnerStateT], AnakinTrainOutput] | None,
    dict[str, Any],  # networks
    LearnerStateT | None,
]:
    logger_prefix = (
        f"GENERIC_SETUP.{build_network_fn.__name__.replace('build_', '').replace('_network', '')}"
    )
    setup_logger = logging.getLogger(f"{logger.name}.{logger_prefix}")
    setup_logger.debug(f"Setup with_params={with_params}, num_scan={num_updates_per_scan}")

    env = global_args.env
    config = global_args.config
    key_main, _, *net_keys_list = global_args.algo_specific_keys

    network_setup_spec = build_network_setup_fn()
    # Keys are needed for both network specs and standalone param specs
    all_param_spec_names = list(network_setup_spec.network_specs.keys()) + list(
        network_setup_spec.param_specs.keys()
    )
    net_keys_for_init = (
        {
            spec_name: key
            for spec_name, key in zip(all_param_spec_names, net_keys_list, strict=False)
        }
        if with_params
        else {}
    )

    nets_opts_mode = "give_me_fns_and_init_params" if with_params else "give_me_only_fns"
    nets_and_opts_result = build_networks_and_optimizers(
        network_setup=network_setup_spec,
        env=env,
        config=config,
        network_keys=net_keys_for_init,
        hyperparam_batch=next(iter(hyperparam_batch_wrappers.values())),
        instantiate_fn=instantiate_from_config,
        network_builder_fn=build_network_fn,
        optimizer_builder_fn=build_optimizer_fn,
        mode=nets_opts_mode,
    )
    networks = nets_and_opts_result.networks

    core_single_cycle_fn = build_update_step_fn(
        env=env,
        nets_and_opts=nets_and_opts_result,
        normalizer_fns=global_args.normalizer_fns,
        config=config,
        train_strategy=train_strategy,
        hyperparam_batch_wrappers=hyperparam_batch_wrappers,
        hyperparam_non_vectorizeds=hyperparam_non_vectorizeds,
    )

    single_step_core_transformed_fn, scanned_core_transformed_fn = build_scanned_learner_wrapper(
        core_single_cycle_fn, num_updates_per_scan
    )
    distributed_single_fn = transform_function_by_strategy(
        single_step_core_transformed_fn, train_strategy, config.training.jit_enabled
    )
    distributed_scanned_fn = (
        transform_function_by_strategy(
            scanned_core_transformed_fn, train_strategy, config.training.jit_enabled
        )
        if scanned_core_transformed_fn
        else None
    )

    initial_learner_state: LearnerStateT | None = None
    if with_params:
        initial_learner_state = build_distributed_layout_fn(
            env=env,
            nets_and_opts=nets_and_opts_result,
            hyperparam_batch_wrappers=hyperparam_batch_wrappers,
            normalizer_fns=global_args.normalizer_fns,
            key=key_main,
            train_strategy=train_strategy,
            config=config,
        )

    if with_params and build_warmup_rollout_fn:
        from hyperlax.layout.data import distribute_keys_across_axes

        core_warmup_fn = build_warmup_rollout_fn(
            env,
            nets_and_opts_result,
            global_args.normalizer_fns,
            hyperparam_batch_wrappers,
            config,
        )
        distributed_warmup_fn = transform_function_by_strategy(
            core_warmup_fn, train_strategy, config.training.jit_enabled
        )
        warmup_keys_dist, _ = distribute_keys_across_axes(key_main, train_strategy)

        (
            env_states_wu,
            timesteps_wu,
            key_wu,
            buffer_states_wu,
            norm_params_wu,
            env_steps_wu,
        ) = distributed_warmup_fn(
            initial_learner_state.env_state,
            initial_learner_state.timestep,
            initial_learner_state.buffer_state,
            warmup_keys_dist,
            initial_learner_state.normalization_params,
            initial_learner_state.algo_hyperparams,
        )

        initial_learner_state = initial_learner_state._replace(
            env_state=env_states_wu,
            timestep=timesteps_wu,
            key=key_wu,
            buffer_state=buffer_states_wu,
            normalization_params=norm_params_wu,
            total_env_steps_counter=env_steps_wu,
        )

    return (
        distributed_single_fn,
        distributed_scanned_fn,
        networks,
        initial_learner_state,
    )


def setup_observers(
    config,
    initial_num_hyperparams,
    target_total_steps,
    num_evaluation_milestones,
    system_monitor,
):
    observers = []
    if config.logger.enable_hyperparam_progress_bar:
        progress_observer = ProgressDisplayObserver(
            initial_num_hyperparams=initial_num_hyperparams,
            target_total_steps=target_total_steps,
            all_possible_milestones=_calculate_milestones(
                target_total_steps, num_evaluation_milestones
            ),
            display_fn=display_hp_progress,
            system_monitor=system_monitor,
        )
        observers.append(progress_observer)
        logger.info("Added hyperparam progress bar to observers.")
    return observers


def jit_gpu_flags(config):
    enable_jit = (
        getattr(config, "console_logger", None)
        and getattr(config.console_logger, "level", "").upper() == "DEBUG"
        and getattr(config, "logger", None)
        and getattr(config.logger, "block_until_enabled", False)
    )
    enable_gpu = (
        getattr(config, "console_logger", None)
        and getattr(config.console_logger, "level", "").upper() == "DEBUG"
        and getattr(config, "logger", None)
        and getattr(config.logger, "enable_gpu_memory_logging", False)
    )
    return enable_jit, enable_gpu


def select_train_style_fn(style, single_fn, scanned_fn):
    if style == "phased_scan_step":
        logger.debug(f"Using SCANNED train_fn for trainer_style: {style}")
        return scanned_fn
    else:
        logger.debug(f"Using SINGLE_STEP train_fn for trainer_style: {style}")
        return single_fn


def build_scanned_learner_wrapper(
    single_cycle_fn: Callable[[LearnerStateT, Any], tuple[LearnerStateT, AnakinTrainOutput]],
    num_updates_per_scan: int,
    metrics_aggregation_strategy: str = "last",
) -> tuple[
    Callable[[LearnerStateT], AnakinTrainOutput],
    Callable[[LearnerStateT], AnakinTrainOutput],
]:
    """
    Explicit version where you pass the output constructor directly.
    """

    def _aggregate_metrics(per_cycle_outputs, strategy: str):
        if strategy == "last":
            return jax.tree_util.tree_map(lambda x: x[-1], per_cycle_outputs)
        elif strategy == "mean":
            return jax.tree_util.tree_map(lambda x: jnp.mean(x, axis=0), per_cycle_outputs)
        elif strategy == "sum":
            return jax.tree_util.tree_map(lambda x: jnp.sum(x, axis=0), per_cycle_outputs)
        else:
            raise ValueError(f"Unknown aggregation strategy: {strategy}")

    def scanned_learner_fn(initial_learner_state: LearnerStateT) -> AnakinTrainOutput:
        final_learner_state, per_cycle_outputs = jax.lax.scan(
            single_cycle_fn, initial_learner_state, None, length=num_updates_per_scan
        )

        # Aggregate metrics
        aggregated_episode_metrics = _aggregate_metrics(
            per_cycle_outputs.episode_metrics, metrics_aggregation_strategy
        )
        aggregated_train_metrics = _aggregate_metrics(
            per_cycle_outputs.train_metrics, metrics_aggregation_strategy
        )

        # Use the explicit constructor
        return AnakinTrainOutput(
            learner_state=final_learner_state,
            episode_metrics=aggregated_episode_metrics,
            train_metrics=aggregated_train_metrics,
        )

    def single_step_learner_fn(
        initial_learner_state: LearnerStateT,
    ) -> AnakinTrainOutput:
        final_state, cycle_output = single_cycle_fn(initial_learner_state, None)
        return cycle_output

    return single_step_learner_fn, scanned_learner_fn
