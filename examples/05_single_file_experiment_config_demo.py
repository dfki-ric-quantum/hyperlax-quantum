"""
A single, self-contained, and runnable script demonstrating how to configure and
launch hyperlax experiments programmatically.

This file is designed for newcomers and for users who prefer a script-based workflow.
It provides a clear separation between user-configurable settings and the underlying
framework implementation.

================================================================================
                                  HOW TO USE
================================================================================

This script is divided into two main areas:

  1. THE USER ZONE (SECTION 1):
     This is the ONLY section you need to modify for 99% of use cases.
     - To change default hyperparameters for a single run, edit `HyperparameterDefaults`.
     - To define a hyperparameter sweep, edit the `get_search_space()` dictionary.
     - To change training duration, edit `ExperimentSettings`.

  2. THE FRAMEWORK ZONE (SECTIONS 2, 3, 4):
     This contains the core logic that makes the experiment work. You should not
     need to modify anything in these sections unless you are an advanced user
     extending the framework.

--------------------------------------------------------------------------------

To run the script from your terminal:

  - To run a single experiment with default settings:
    python -m hyperlax.examples.single_file_experiment_config_demo single

  - To run a batched hyperparameter sweep:
    python -m hyperlax.examples.single_file_experiment_config_demo sweep
"""

# --- Standard Library Imports ---
import dataclasses
import logging
import sys
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

# --- Third-Party Imports ---
import pandas as pd
from tabulate import tabulate

# --- hyperlax Project Imports ---
from hyperlax.algo.ppo.main_ppo import main as ppo_main
from hyperlax.configs.main_base import BaseExperimentConfig
from hyperlax.configs.mapping import apply_distributions_to_config
from hyperlax.configs.modifiers.common_settings import apply_quick_test_settings
from hyperlax.hyperparam.base_types import flatten_tunables
from hyperlax.hyperparam.distributions import Categorical, LogUniform
from hyperlax.hyperparam.sampler import (
    SamplingConfig,
    _apply_joint_sampling_rules,
    generate_independent_samples,
)
from hyperlax.hyperparam.tunable import Tunable
from hyperlax.logger.console import configure_global_logging
from hyperlax.logger.metrics import summarize_hyperparam_performance
from hyperlax.logger.return_tracker import load_hyperparam_returns_as_named_tuples
from hyperlax.runner.base_types import AlgoSpecificExperimentConfigContainer
from hyperlax.runner.batch_utils import find_sample_id_key
from hyperlax.runner.launcher_utils import _get_ordered_vectorized_keys

logger = logging.getLogger(__name__)

# ==============================================================================
# SECTION 1: USER-CONFIGURABLE SETTINGS
# ==============================================================================
# This is the primary area for modification. Define your experiment defaults
# and search spaces here using simple Python types. Below are examples of
# common modifications you might want to make.
# ==============================================================================


@dataclass(frozen=True)
class ExperimentSettings:
    """
    High-level settings for the experiment runs.
    Modify these values to control the duration and scale of your experiments.

    --- Example: A quick debug run ---
    # To run a very short test to ensure everything is working, you could use:
    # single_run_timesteps: int = 4_000
    # sweep_timesteps_per_sample: int = 10_000
    # sweep_num_samples: int = 4
    """

    # --- General Settings ---
    seed: int = 42

    # --- Single Run Settings ---
    # Total environment steps for a single experiment run.
    single_run_timesteps: int = int(5e6)

    # --- Sweep Settings ---
    # Total environment steps FOR EACH hyperparameter sample in a sweep.
    sweep_timesteps_per_sample: int = int(1e7)
    # The total number of hyperparameter combinations to sample and run.
    sweep_num_samples: int = 32


@dataclass(frozen=True)
class HyperparameterDefaults:
    """
    Default values for a SINGLE PPO run.
    This acts as the "blueprint" configuration. When you run in 'single' mode,
    these are the values that will be used. When you run a 'sweep', any parameter
    NOT defined in `get_search_space()` will fall back to the value here.

    --- Example: Testing a different learning rate or network size ---
    # You can change any value here for a quick test. For instance:
    # actor_lr: float = 1e-2  # A much higher learning rate
    # layer_sizes: list[int] = field(default_factory=lambda: [256, 256]) # A bigger network
    #
    # Alternatively, you can programmatically override these in the `if __name__ == "__main__"`
    # block at the bottom of the script, which is useful for scripting.
    """

    actor_lr: float = 3e-3
    critic_lr: float = 3e-4
    gamma: float = 0.98
    gae_lambda: float = 0.93
    clip_eps: float = 0.4
    ent_coef: float = 0.002
    vf_coef: float = 0.6
    max_grad_norm: float = 0.4
    rollout_length: int = 2
    epochs: int = 2
    num_minibatches: int = 64
    total_num_envs: int = 2048
    standardize_advantages: bool = True
    decay_learning_rates: bool = True
    normalize_observations: bool = True
    # Network Architecture
    layer_sizes: list[int] = field(default_factory=lambda: [32, 32])
    activation: str = "silu"
    use_layer_norm: bool = False


def get_search_space() -> dict[str, Any]:
    """
    Defines the search space for a hyperparameter sweep.
    To use a different example, comment out the active `return` statement
    and uncomment the one you want to use.

    - Keys are the full path to the hyperparameter.
    - Values are hyperlax distribution objects (e.g., LogUniform, Categorical).
    """

    # --- Example 1: A broad sweep over learning rates and network architecture ---
    return {
        "algorithm.hyperparam.actor_lr": LogUniform(domain=(1e-5, 1e-3)),
        "algorithm.hyperparam.critic_lr": LogUniform(domain=(1e-5, 1e-3)),
        "algorithm.hyperparam.ent_coef": LogUniform(domain=(1e-4, 1e-2)),
        "algorithm.network.actor_network.pre_torso.layer_sizes": Categorical(
            values=[[32, 32], [64, 64], [128, 128]]
        ),
        "algorithm.network.actor_network.pre_torso.activation": Categorical(
            values=["silu", "relu", "tanh"]
        ),
    }

    # --- Example 2: A smaller, focused sweep on just PPO coefficients ---
    # return {
    #     "algorithm.hyperparam.gamma": Categorical(values=[0.99, 0.995, 0.999]),
    #     "algorithm.hyperparam.gae_lambda": Categorical(values=[0.9, 0.95, 1.0]),
    #     "algorithm.hyperparam.clip_eps": Categorical(values=[0.1, 0.2, 0.3]),
    #     "algorithm.hyperparam.ent_coef": LogUniform(domain=(1e-4, 1e-2)),
    # }

    # --- Example 3: A sweep to find the best network size for a fixed learning rate ---
    # Note: actor_lr is NOT included here, so it will use the default value from HyperparameterDefaults.
    # return {
    #     "algorithm.network.actor_network.pre_torso.layer_sizes": Categorical(
    #         values=[[16], [32, 32], [64, 64, 64]]
    #     ),
    #     "algorithm.network.actor_network.pre_torso.activation": Categorical(
    #         values=["silu", "tanh"]
    #     ),
    #     "algorithm.network.actor_network.pre_torso.use_layer_norm": Categorical(
    #         values=[True, False]
    #     ),
    # }

# ==============================================================================
# SECTION 2: CORE EXPERIMENT BLUEPRINT
# ==============================================================================
# This section defines the structure of the experiment configuration using the
# framework's internal =Tunable= objects.
#
# !!! ADVANCED USERS ONLY: Do NOT modify this section unless you understand
# !!! the internal mechanics of the hyperlax runner.
# ==============================================================================

# --- Dataclasses defining the configuration structure. ---
# --- DO NOT MODIFY `is_vectorized`, `is_fixed`, `expected_type` ---

@dataclass(frozen=True)
class LoggerConfig:
    """Configuration for logging and experiment output."""

    enabled: bool = True
    level: str = "INFO"
    show_timestamp: bool = True
    save_console_to_file: bool = True
    console_log_filename: str = "hyperlax.log"
    base_exp_path: str = "results/ppo_mlp_single_file"
    aggregate_metrics: bool = True
    enable_hyperparam_progress_bar: bool = True
    enable_jax_debug_prints: bool = False
    enable_timing_logs: bool = False
    enable_gpu_memory_logging: bool = False
    checkpointing_enabled: bool = False
    enable_summarize_layout: bool = False
    initialized: bool = True


@dataclass(frozen=True)
class BaseTrainingConfig:
    """Configuration for the training loop and evaluation process."""

    seed: int = 42
    total_timesteps: int = int(5e6)
    num_evaluation: int = 20
    num_eval_episodes: int = 32
    evaluation_greedy: bool = False
    num_agents_slash_seeds: int = 2
    jit_enabled: bool = True
    normalize_method: str = "running_meanstd"
    hyperparam_batch_enabled: bool = False
    hyperparam_batch_samples: dict[str, list[list[Any]]] = field(default_factory=dict)
    hyperparam_batch_size: int = -1
    hyperparam_batch_sample_ids: list[int] = field(default_factory=list)
    trainer_style: str = "phased"
    update_batch_size: int = 1
    num_devices: int = -1


@dataclass(frozen=True)
class ScenarioConfig:
    """Defines the specific task within an environment family."""

    name: str = "Pendulum-v1"
    task_name: str = "pendulum"


@dataclass(frozen=True)
class GymnaxPendulumConfig:
    """Configuration for the Gymnax Pendulum environment."""

    env_name: str = "gymnax.pendulum"
    scenario: ScenarioConfig = field(default_factory=ScenarioConfig)
    kwargs: dict[str, Any] = field(default_factory=dict)
    eval_metric: str = "episode_return"
    obs_dim: int = -1
    act_dim: int = -1
    act_minimum: float = -2.0
    act_maximum: float = 2.0


@dataclass(frozen=True)
class MLPTorso:
    """Configuration for a standard Multi-Layer Perceptron (MLP) torso."""

    _target_: str = "hyperlax.network.torso.MLPTorso"
    layer_sizes: Tunable = field(init=False)
    activation: Tunable = field(init=False)
    use_layer_norm: Tunable = field(init=False)


@dataclass(frozen=True)
class ActionHead:
    """Config for the actor's output layer. Patched by the runner for the env."""

    _target_: str = "hyperlax.network.heads.NormalAffineTanhDistributionHead"
    minimum: float = -1.0
    maximum: float = 1.0


@dataclass(frozen=True)
class CriticHead:
    """Config for the critic's output layer."""

    _target_: str = "hyperlax.network.heads.ScalarCriticHead"


@dataclass(frozen=True)
class MLPActorNetworkConfig:
    pre_torso: MLPTorso = field(default_factory=MLPTorso)
    action_head: ActionHead = field(default_factory=ActionHead)


@dataclass(frozen=True)
class MLPCriticNetworkConfig:
    pre_torso: MLPTorso = field(default_factory=MLPTorso)
    critic_head: CriticHead = field(default_factory=CriticHead)


@dataclass(frozen=True)
class MLPActorCriticConfig:
    actor_network: MLPActorNetworkConfig = field(default_factory=MLPActorNetworkConfig)
    critic_network: MLPCriticNetworkConfig = field(default_factory=MLPCriticNetworkConfig)


@dataclass(frozen=True)
class PPOHyperparams:
    """Structure for PPO hyperparameters using Tunable for the framework."""

    actor_lr: Tunable = field(init=False)
    critic_lr: Tunable = field(init=False)
    gamma: Tunable = field(init=False)
    gae_lambda: Tunable = field(init=False)
    clip_eps: Tunable = field(init=False)
    ent_coef: Tunable = field(init=False)
    vf_coef: Tunable = field(init=False)
    max_grad_norm: Tunable = field(init=False)
    rollout_length: Tunable = field(init=False)
    epochs: Tunable = field(init=False)
    num_minibatches: Tunable = field(init=False)
    total_num_envs: Tunable = field(init=False)
    standardize_advantages: Tunable = field(init=False)
    decay_learning_rates: Tunable = field(init=False)
    normalize_observations: Tunable = field(init=False)
    sample_id: Tunable = field(init=False)  # Internal framework field


def get_experiment_blueprint(defaults: HyperparameterDefaults) -> BaseExperimentConfig:
    """
    Internal factory to construct the full experiment config object.
    It translates the simple defaults from SECTION 1 into the complex =Tunable=
    objects required by the framework's backend.
    """
    # This function acts as a bridge, creating the complex objects the framework
    # currently needs, based on the simple user-defined defaults.

    # Define internal specifications for each Tunable parameter
    hp_specs = {
        "actor_lr": {"is_vectorized": True, "type": float},
        "critic_lr": {"is_vectorized": True, "type": float},
        "gamma": {"is_vectorized": True, "type": float},
        "gae_lambda": {"is_vectorized": True, "type": float},
        "clip_eps": {"is_vectorized": True, "type": float},
        "ent_coef": {"is_vectorized": True, "type": float},
        "vf_coef": {"is_vectorized": True, "type": float},
        "max_grad_norm": {"is_vectorized": True, "type": float},
        "rollout_length": {"is_vectorized": True, "type": int},
        "epochs": {"is_vectorized": True, "type": int},
        "num_minibatches": {"is_vectorized": True, "type": int},
        "total_num_envs": {"is_vectorized": True, "type": int},
        "standardize_advantages": {"is_vectorized": True, "type": bool},
        "decay_learning_rates": {"is_vectorized": True, "type": bool},
        "normalize_observations": {"is_vectorized": True, "type": bool},
        "sample_id": {"is_vectorized": True, "type": int, "value": -1},  # Special internal value
    }
    net_specs = {
        "layer_sizes": {"is_vectorized": False, "type": list[int]},
        "activation": {"is_vectorized": False, "type": str},
        "use_layer_norm": {"is_vectorized": False, "type": bool},
    }

    # Create PPOHyperparams object
    ppo_hps = PPOHyperparams()
    for name, spec in hp_specs.items():
        if "value" in spec:
            val = spec["value"]  # use internal default
        else:
            val = getattr(defaults, name)  # get from user defaults if it exists

        tunable = Tunable(
            value=val,
            is_vectorized=spec["is_vectorized"],
            is_fixed=True,
            expected_type=spec["type"],
        )
        object.__setattr__(ppo_hps, name, tunable)

    # Create MLPTorso object
    mlp_torso = MLPTorso()
    for name, spec in net_specs.items():
        tunable = Tunable(
            value=getattr(defaults, name),
            is_vectorized=spec["is_vectorized"],
            is_fixed=True,
            expected_type=spec["type"],
        )
        object.__setattr__(mlp_torso, name, tunable)

    # Assemble the full config
    actor_network_cfg = MLPActorNetworkConfig(pre_torso=mlp_torso)
    critic_network_cfg = MLPCriticNetworkConfig(pre_torso=mlp_torso)
    actor_critic_cfg = MLPActorCriticConfig(
        actor_network=actor_network_cfg, critic_network=critic_network_cfg
    )
    ppo_cfg = dataclasses.make_dataclass(
        "PPOConfig", [("_target_", str), ("network", Any), ("hyperparam", PPOHyperparams)]
    )(
        _target_="hyperlax.algo.ppo.main_ppo.main",
        network=actor_critic_cfg,
        hyperparam=ppo_hps,
    )

    return BaseExperimentConfig(
        algorithm=ppo_cfg,
        env=GymnaxPendulumConfig(),
        training=BaseTrainingConfig(),
        logger=LoggerConfig(),
        config_name="ppo_mlp_single_file_example",
    )


# ==============================================================================
# SECTION 3: HIGH-LEVEL LAUNCHERS & ANALYSIS
# ==============================================================================
# These functions orchestrate the experiment runs. They use the settings from
# SECTION 1 and the blueprint from SECTION 2.
# ==============================================================================


def launch_single_experiment(
    settings: ExperimentSettings,
    defaults: HyperparameterDefaults,
    output_dir: str,
    overrides: dict[str, Any] | None = None,
    quick_run: bool = False,
) -> None:
    """Configures and launches a single PPO experiment."""
    if overrides:
        logger.info(f"Applying overrides to defaults: {overrides}")
        defaults = dataclasses.replace(defaults, **overrides)

    base_config = get_experiment_blueprint(defaults)
    run_config = _prepare_config_for_single_run(
        base_config, output_dir, settings.single_run_timesteps, quick_run
    )

    configure_global_logging(run_config.logger, overall_log_prefix="PPO_SINGLE_DEMO")

    print("\n--- Hyperparameters for Single Run ---")
    hp_data = [[f"algorithm.hyperparam.{k}", v] for k, v in dataclasses.asdict(defaults).items() if k in [f.name for f in dataclasses.fields(PPOHyperparams)]]
    net_data = [[f"algorithm.network...{k}", v] for k,v in dataclasses.asdict(defaults).items() if k in [f.name for f in dataclasses.fields(MLPTorso)]]

    print("\n" + tabulate(hp_data + net_data, headers=["Parameter", "Value"], tablefmt="pipe"))

    print(f"\nLaunching single PPO run. Results will be in: {run_config.logger.base_exp_path}")
    ppo_main(run_config)
    print("\n--- Single Experiment Demo Finished ---")

    _analyze_and_print_summary(output_dir, sort_by="peak_performance")


def launch_hyperparameter_sweep(
    settings: ExperimentSettings,
    defaults: HyperparameterDefaults,
    search_space: dict[str, Any],
    output_dir: str,
    quick_run: bool = False,
) -> None:
    """Configures and launches a batched PPO hyperparameter sweep."""
    base_config = get_experiment_blueprint(defaults)
    run_config = _prepare_config_for_sweep(
        base_config,
        output_dir,
        settings.sweep_timesteps_per_sample,
        settings.sweep_num_samples,
        search_space,
        quick_run,
    )

    configure_global_logging(run_config.logger, overall_log_prefix="PPO_SWEEP_DEMO")

    print(f"\nLaunching batched PPO sweep with {settings.sweep_num_samples} total HPs.")
    print(f"Results will be in: {run_config.logger.base_exp_path}")
    ppo_main(run_config)
    print("\n--- Batched Sweep Demo Finished ---")

    _analyze_and_print_summary(output_dir, sort_by="total_score")


def _analyze_and_print_summary(output_dir: str, sort_by: str = "peak_performance") -> None:
    """Loads results from an experiment directory and prints a performance summary."""
    print("\n--- Experiment Analysis Summary ---")
    results_file = Path(output_dir) / "return_group_by_hyperparams.npz"
    if not results_file.exists():
        print(f"Results file not found: {results_file}")
        return
    try:
        return_trackers = load_hyperparam_returns_as_named_tuples(results_file)
        if not return_trackers:
            print("No data found in results file.")
            return
        summarize_hyperparam_performance(return_trackers, top_n=10, sort_by=sort_by)
    except Exception as e:
        print(f"An error occurred during analysis: {e}")


# ==============================================================================
# SECTION 4: FRAMEWORK INTERNALS & HELPERS
# ==============================================================================
# The functions below contain the complex logic required to translate the user
# settings into a format the hyperlax runner can execute.
#
# !!! ADVANCED USERS ONLY: Do NOT modify this section unless you are
# !!! familiar with the framework's internal data structures.
# ==============================================================================


def _prepare_config_for_single_run(
    base_config: BaseExperimentConfig,
    output_dir: str,
    total_timesteps: int,
    quick_run: bool = False,
) -> BaseExperimentConfig:
    """Creates a ready-to-run config for a single experiment."""
    config = dataclasses.replace(
        base_config, logger=dataclasses.replace(base_config.logger, base_exp_path=output_dir)
    )
    config.training = dataclasses.replace(config.training, total_timesteps=total_timesteps)

    if quick_run:
        logger.info("Applying quick run settings for a short test.")
        config = apply_quick_test_settings(config)

    # This crucial step packages the single set of HPs into a "batch of size 1"
    flat_tunables = flatten_tunables(config.algorithm)
    all_defaults = {f"algorithm.{path}": spec.value for path, spec in flat_tunables.items()}
    vec_arrays = {}
    component_paths = {
        "algo": "algorithm.hyperparam",
        "network_actor": "algorithm.network.actor_network",
        "network_critic": "algorithm.network.critic_network",
    }
    for name, path in component_paths.items():
        component = config
        for part in path.split("."):
            component = getattr(component, part)
        vec_keys = _get_ordered_vectorized_keys(component, path)
        if vec_keys:
            row = [all_defaults.get(key) for key in vec_keys]
            vec_arrays[name] = [row]

    config.training = dataclasses.replace(
        config.training,
        hyperparam_batch_enabled=True,
        hyperparam_batch_size=1,
        hyperparam_batch_samples=vec_arrays,
        hyperparam_batch_sample_ids=[0],
    )
    return config


def _prepare_config_for_sweep(
    base_config: BaseExperimentConfig,
    output_dir: str,
    total_timesteps: int,
    num_samples: int,
    search_space: dict[str, Any],
    quick_run: bool = False,
) -> BaseExperimentConfig:
    """Creates a ready-to-run config for a batched hyperparameter sweep."""
    config = dataclasses.replace(
        base_config, logger=dataclasses.replace(base_config.logger, base_exp_path=output_dir)
    )
    config.training = dataclasses.replace(config.training, total_timesteps=total_timesteps)
    if quick_run:
        logger.info("Applying quick run settings for a short test.")
        config = apply_quick_test_settings(config)

    config = apply_distributions_to_config(config, search_space)

    exp_container = AlgoSpecificExperimentConfigContainer("PPO", ppo_main, config, search_space, None)
    sampling_cfg = SamplingConfig(search_space, num_samples, seed=42)
    sample_id_key = find_sample_id_key(config)
    sampling_result = generate_independent_samples(sampling_cfg, sample_id_key)
    samples_dict = _apply_joint_sampling_rules(sampling_result.unn, exp_container)

    all_defaults = {
        f"algorithm.{path}": spec.value for path, spec in flatten_tunables(config.algorithm).items()
    }
    vec_arrays = {}
    component_paths = {
        "algo": "algorithm.hyperparam",
        "network_actor": "algorithm.network.actor_network",
        "network_critic": "algorithm.network.critic_network",
    }
    for name, path in component_paths.items():
        component = config
        for part in path.split("."):
            component = getattr(component, part)
        vec_keys = _get_ordered_vectorized_keys(component, path)
        if vec_keys:
            rows = []
            for i in range(num_samples):
                row = [samples_dict.get(key, [all_defaults.get(key)])[i] for key in vec_keys]
                rows.append(row)
            vec_arrays[name] = rows

    config.training = dataclasses.replace(
        config.training,
        hyperparam_batch_enabled=True,
        hyperparam_batch_size=num_samples,
        hyperparam_batch_samples=vec_arrays,
        hyperparam_batch_sample_ids=samples_dict[sample_id_key],
    )
    return config


# ==============================================================================
# SECTION 5: SCRIPT ENTRYPOINT
# ==============================================================================

if __name__ == "__main__":
    # Get user-defined settings from SECTION 1
    exp_settings = ExperimentSettings()
    hp_defaults = HyperparameterDefaults()

    if len(sys.argv) > 1 and sys.argv[1] == "single":
        # ----------------------------------------------------------------------
        # EXAMPLE: Programmatically overriding defaults for a single run.
        # This is useful for scripting or quick, one-off tests without
        # editing the dataclasses above. Just uncomment and modify the dict.
        # ----------------------------------------------------------------------
        custom_overrides = {
            # "actor_lr": 5e-3,
            # "layer_sizes": [128, 128],
            # "activation": "tanh",
        }
        # If the dictionary is empty, the script will just use the defaults.
        # ----------------------------------------------------------------------

        launch_single_experiment(
            settings=exp_settings,
            defaults=hp_defaults,
            output_dir="results/ppo_single_run_demo",
            overrides=custom_overrides,
        )
    elif len(sys.argv) > 1 and sys.argv[1] == "sweep":
        search_space_for_sweep = get_search_space()
        launch_hyperparameter_sweep(
            settings=exp_settings,
            defaults=hp_defaults,
            search_space=search_space_for_sweep,
            output_dir="results/ppo_sweep_demo",
        )
    else:
        print("Usage: python -m hyperlax.examples.single_file_experiment_config_demo [single|sweep]")
        print("\nRunning 'single' mode by default...")
        launch_single_experiment(
            settings=exp_settings,
            defaults=hp_defaults,
            output_dir="results/ppo_single_run_demo",
        )
