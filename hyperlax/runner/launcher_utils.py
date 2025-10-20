import copy
import importlib
import logging
import os
import re
import tempfile
from pathlib import Path
from typing import Any

from hyperlax.algo.dqn.config import DQNConfig
from hyperlax.algo.sac.config import SACConfig
from hyperlax.configs.logger.base_logger import LoggerConfig
from hyperlax.configs.main_base import (BaseExperimentConfig,
                                        get_sampling_distributions_from_config)
from hyperlax.configs.mapping import apply_distributions_to_config
from hyperlax.configs.modifiers.common_settings import (
    apply_long_run_settings, apply_quick_test_settings,
    apply_runtime_benchmark_settings,
    apply_runtime_benchmark_settings_given_timesteps)
from hyperlax.env.make_env import make as make_env
from hyperlax.logger.console import configure_global_logging
from hyperlax.network.parametric_torso import ACTIVATION_FN_TO_IDX
from hyperlax.runner.base_types import AlgoSpecificExperimentConfigContainer
from hyperlax.runner.batch_utils import find_sample_id_key
from hyperlax.runner.launch_args import BenchmarkRunConfig
from hyperlax.runner.sampling import (_get_default_values_from_config,
                                      _get_ordered_vectorized_keys)
from hyperlax.utils.algo_setup import (extract_environment_info,
                                       update_config_with_env_info)


logger = logging.getLogger(__name__)


def load_benchmark_config(config_name: str) -> BenchmarkRunConfig:
    """Dynamically loads a benchmark configuration module."""
    logger.debug(f"Loading benchmark config: {config_name}")
    module_path = f"hyperlax.configs.benchmark.{config_name}"
    try:
        module = importlib.import_module(module_path)
        if not hasattr(module, "get_benchmark_config"):
            raise AttributeError(
                f"Module '{module_path}' must have a 'get_benchmark_config' function."
            )

        config = module.get_benchmark_config()
        if not isinstance(config, BenchmarkRunConfig):
            raise TypeError(
                f"'get_benchmark_config' in '{module_path}' must return an instance of BenchmarkConfig."
            )
        logger.debug(f"Successfully loaded benchmark config: {config_name}")
        return config
    except ImportError:
        logger.error(f"Could not import benchmark config module: {module_path}")
        raise
    except (AttributeError, TypeError) as e:
        logger.error(f"Error loading benchmark config '{config_name}': {e}")
        raise


# --- Logging Setup ---
def setup_logging(log_level: str):
    """Configures global logging for the application."""
    log_level = log_level.upper()
    configure_global_logging(
        LoggerConfig(level=log_level, initialized=True),
        show_header=True,
        overall_log_prefix="HYPERLAX",
    )


# --- Output Directory Setup ---
def sanitize_for_path(s: str) -> str:
    """Sanitizes a string for use in a file path."""
    return re.sub(
        r"[^a-z0-9_]",
        "",
        s.lower().replace(".", "_").replace("/", "_").replace(" ", "_"),
    )


def get_tmp_dir() -> str:
    """Creates a temporary directory for a run."""
    import datetime

    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    tmp_path = os.path.join(tempfile.gettempdir(), f"hyperlax_run_{timestamp}")
    os.makedirs(tmp_path, exist_ok=True)
    return tmp_path


def setup_output_directory(args) -> Path:
    """Determines the final output directory for the experiment."""
    output_root = Path(args.output_root) if args.output_root else Path(get_tmp_dir())

    # If resuming, the output_root is already the final experiment directory.
    if hasattr(args, "resume") and args.resume:
        output_root.mkdir(parents=True, exist_ok=True)
        return output_root

    # If the output_root name already contains the algo and env, don't create subdirs.
    # This is for benchmark scripts that pre-create descriptive directories.

    # The check must use the same string format as the directory name generator.
    # launch_benchmark uses hyphens for algos and dots for envs.
    algo_part_for_check = args.algo_and_network_config.replace("_", "-")
    env_part_for_check = (
        args.env_config if hasattr(args, "env_config") and args.env_config else ""
    )

    if (
        algo_part_for_check in output_root.name
        and env_part_for_check
        and env_part_for_check in output_root.name
    ):
        output_root.mkdir(parents=True, exist_ok=True)
        return output_root
    else:
        # Fallback for direct calls to =sweep-hp-samples= etc.
        algo_part_sanitized = sanitize_for_path(args.algo_and_network_config)
        env_part_sanitized = sanitize_for_path(env_part_for_check) if env_part_for_check else ""

        run_dir = output_root / algo_part_sanitized
        if env_part_sanitized:
            run_dir = run_dir / env_part_sanitized
        run_dir.mkdir(parents=True, exist_ok=True)
        return run_dir


def load_env_config(env_name: str) -> Any:
    """
    Load environment configuration based on framework and type.
    Example: 'gymnax.pendulum' will load GymnaxPendulumConfig
             'gymnax.mountain_car_continuous' will load GymnaxMountainCarContinuousConfig
    NOTE Pay attention your naming convention <framework>.<type> -> <Framework><Type>Config
    """
    if not env_name or "." not in env_name:
        raise ValueError(f"Invalid env_name format: '{env_name}'. Expected '<framework>.<type>'.")

    framework, env_type = env_name.split(".", 1)
    module_path = f"hyperlax.configs.env.{framework}.{env_type}"
    try:
        module = importlib.import_module(module_path)
    except ImportError as e:
        raise ImportError(
            f"Could not import environment config module '{module_path}' for env '{env_name}': {str(e)}"
        )

    clean_framework = re.sub(r"[^a-zA-Z0-9]", "", framework)
    clean_env_type = re.sub(r"[^a-zA-Z0-9]", "", env_type)
    target_pattern = f"{clean_framework}{clean_env_type}config".lower()

    for name, obj in module.__dict__.items():
        if isinstance(obj, type) and name.lower().endswith("config"):
            clean_class_name = re.sub(r"[^a-zA-Z0-9]", "", name).lower()
            if clean_class_name == target_pattern:
                return obj()

    raise ValueError(
        f"No matching config class found for '{env_name}' in module '{module_path}'. "
        f"Expected a class matching pattern: *{target_pattern}"
    )


def build_main_experiment_config(
    args: Any, skip_env_setup: bool = False
) -> AlgoSpecificExperimentConfigContainer:
    """
    Builds the complete experiment configuration container from CLI arguments.
    """
    # 1. Load base recipe
    module_path = f"hyperlax.configs.algo.{args.algo_and_network_config}"
    try:
        module = importlib.import_module(module_path)
    except ImportError:
        logger.error(f"Could not import base recipe module: {module_path}")
        raise

    base_exp_config: BaseExperimentConfig = module.get_base_config()
    current_exp_config = copy.deepcopy(base_exp_config)

    # 2. Select and apply the correct hyperparameter distributions first.
    dist_dict = {}
    if args.run_length_modifier == "quick" and hasattr(
        module, "get_base_hyperparam_distributions_test_quick"
    ):
        logger.info("Using 'quick test' hyperparameter distributions.")
        dist_dict = module.get_base_hyperparam_distributions_test_quick()
    elif hasattr(module, "get_base_hyperparam_distributions"):
        logger.info("Using standard hyperparameter distributions.")
        dist_dict = module.get_base_hyperparam_distributions()

    if dist_dict:
        current_exp_config = apply_distributions_to_config(current_exp_config, dist_dict)

    # 3. Apply run-length modifiers. This will modify training settings AND
    #    the default `.value` of Tunable objects, but not their distributions.
    if args.run_length_modifier == "quick":
        current_exp_config = apply_quick_test_settings(current_exp_config)
    elif args.run_length_modifier == "long":
        current_exp_config = apply_long_run_settings(current_exp_config)
    elif args.run_length_modifier == "runtime_benchmark":
        current_exp_config = apply_runtime_benchmark_settings(current_exp_config)
    elif args.run_length_modifier == "runtime_benchmark_drpqc":
        current_exp_config = apply_runtime_benchmark_settings(current_exp_config, pqc_mode=True)
    elif isinstance(args.run_length_modifier, int) or isinstance(args.run_length_modifier, float):
        current_exp_config = apply_runtime_benchmark_settings_given_timesteps(current_exp_config, timesteps=int(args.run_length_modifier))

    # 4. Set environment and perform environment-specific patching
    if hasattr(args, "env_config") and args.env_config and not skip_env_setup:
        current_exp_config.env = load_env_config(args.env_config)
        temp_cfg = copy.deepcopy(current_exp_config)
        if not hasattr(temp_cfg.training, "num_agents_slash_seeds"):
            temp_cfg.training.num_agents_slash_seeds = 1
        env_instance, _ = make_env(config=temp_cfg)
        current_exp_config = update_config_with_env_info(current_exp_config, env_instance)

        # Algorithm-specific compatibility checks
        env_info = extract_environment_info(env_instance)
        if isinstance(current_exp_config.algorithm, DQNConfig):
            if env_info.action_space_type == "continuous":
                raise NotImplementedError(
                    f"DQN algorithm is paired with a continuous environment ('{args.env_config}'). "
                    "This implementation of DQN only supports discrete action spaces."
                )
        elif isinstance(current_exp_config.algorithm, SACConfig):
            if env_info.action_space_type == "discrete":
                raise NotImplementedError(
                    f"SAC algorithm is paired with a discrete environment ('{args.env_config}'). "
                    "This implementation of SAC only supports continuous action spaces."
                )

    # 5. Select the algorithm's main function
    algo_config = current_exp_config.algorithm
    module_path, func_name = algo_config._target_.rsplit(".", 1)
    algo_module = importlib.import_module(module_path)
    algo_main = getattr(algo_module, func_name)
    algo_name = current_exp_config.algorithm.__class__.__name__.replace("Config", "")

    # 6. Extract the final set of distributions for the sampler
    final_hp_dist = get_sampling_distributions_from_config(current_exp_config)
    if "__JOINT_SAMPLING__" in dist_dict:
        final_hp_dist["__JOINT_SAMPLING__"] = dist_dict["__JOINT_SAMPLING__"]

    # 7. Return the fully-packaged container
    return AlgoSpecificExperimentConfigContainer(
        algo_name=algo_name,
        algo_main_fn=algo_main,
        experiment_config=current_exp_config,
        hyperparam_dist_config=final_hp_dist,
        hyperparams_container_spec=None,
    )




def create_single_hp_batch_from_config_defaults(config: BaseExperimentConfig) -> dict[str, list]:
    """
    Creates a batch of size 1 from a config's default Tunable values.
    This is used to package a single experiment run into the batch format expected by the runner.
    """
    all_defaults = _get_default_values_from_config(config)
    vec_arrays = {}

    sample_id = 0
    try:
        # This key is required for tracking and must be defined in the hyperparams.
        sample_id_key = find_sample_id_key(config)
    except ValueError as e:
        logger.error(
            "Could not find a 'Tunable' field for 'sample_id' in the config, "
            "which is required for the runner. Single run setup failed."
        )
        raise e

    def build_row(keys: list[str]) -> list[list[Any]]:
        """Builds a single row for the batch array."""
        row = []
        for key in keys:
            if key == sample_id_key:
                value = sample_id
            else:
                value = all_defaults.get(key)
                if value is None:
                    raise ValueError(
                        f"Could not find a default value for the vectorized key '{key}'"
                    )

            if key.endswith(".activation"):
                str_value = str(value)
                index_value = ACTIVATION_FN_TO_IDX.get(str_value)
                if index_value is None:
                    raise ValueError(
                        f"Unknown activation function name '{str_value}' for key '{key}'"
                    )
                value = index_value
            row.append(value)
        return [row]  # Return as a list containing one row for a batch of size 1

    # Algorithm Hyperparameters
    vec_keys_algo = _get_ordered_vectorized_keys(
        config.algorithm.hyperparam, "algorithm.hyperparam"
    )
    if vec_keys_algo:
        vec_arrays["algo"] = build_row(vec_keys_algo)

    # Actor Network Hyperparameters
    if hasattr(config.algorithm.network, "actor_network"):
        vec_keys_actor = _get_ordered_vectorized_keys(
            config.algorithm.network.actor_network, "algorithm.network.actor_network"
        )
        if vec_keys_actor:
            vec_arrays["network_actor"] = build_row(vec_keys_actor)

    # Critic Network Hyperparameters
    if hasattr(config.algorithm.network, "critic_network"):
        vec_keys_critic = _get_ordered_vectorized_keys(
            config.algorithm.network.critic_network, "algorithm.network.critic_network"
        )
        if vec_keys_critic:
            vec_arrays["network_critic"] = build_row(vec_keys_critic)

    return vec_arrays
