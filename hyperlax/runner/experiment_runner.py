import json
import logging
import time
from dataclasses import replace
from datetime import timedelta
from pathlib import Path
from typing import Any

import jax
import jax.numpy as jnp

from hyperlax.base_types import AlgorithmGlobalSetupArgs
from hyperlax.configs.main_base import BaseExperimentConfig
from hyperlax.env import make_env as environments
from hyperlax.logger.metrics import get_metrics_for_optuna_from_hyperparam_trackers
from hyperlax.logger.return_tracker import save_hyperparam_returns
from hyperlax.logger.sys_monitor import SystemMonitor
from hyperlax.network.hyperparam import (
    MLPVectorizedHyperparams,
)
from hyperlax.normalizer.running_stats import normalizer_setup
from hyperlax.runner.base_types import AlgorithmInterface
from hyperlax.runner.batch_utils import (
    build_hyperparam_batch,
    sort_hyperparams_by_steps,
)
from hyperlax.trainer.phaser import run_training_w_phaser
from hyperlax.utils.algo_setup import setup_observers, update_config_with_env_info

logger = logging.getLogger(__name__)


def run_experiment(
    config: BaseExperimentConfig,
    algo_interface: AlgorithmInterface,
    optuna_objective_names: list[str] = ["peak_performance", "final_performance"],
) -> tuple[list[dict[str, float]], Any, dict[str, float]]:
    """
    Unified entry point for running an algorithm experiment with decoupled hyperparameters.
    """
    start_time = time.time()
    logger.info(f"Starting {algo_interface.algorithm_name_prefix} experiment with decoupled HPs.")

    hp_arrays_dict = config.training.hyperparam_batch_samples
    if "algo" not in hp_arrays_dict:
        raise ValueError("Configuration is missing 'algo' hyperparameter array.")

    # --- Generic Hyperparameter Batch Wrapper Creation ---
    batch_wrappers = {}

    # 1. Build the mandatory 'algo' batch wrapper
    batch_wrappers["algo"] = build_hyperparam_batch(
        array=jnp.array(hp_arrays_dict["algo"]),
        expected_fields=algo_interface.vectorized_hyperparams_cls._fields,
        base_config_component=config.algorithm.hyperparam,
    )
    logger.info(f"Built 'algo' HP batch wrapper. Shape: {batch_wrappers['algo'].shape}")

    # 2. Build network wrappers if they exist. This logic is now fully generic.
    if "network_actor" in hp_arrays_dict:
        batch_wrappers["network_actor"] = build_hyperparam_batch(
            array=jnp.array(hp_arrays_dict["network_actor"]),
            expected_fields=MLPVectorizedHyperparams._fields,  # Assumes MLP for now TODO
            base_config_component=config.algorithm.network.actor_network,
        )
        logger.info(
            f"Built 'network_actor' HP batch wrapper. Shape: {batch_wrappers['network_actor'].shape}"
        )

    if "network_critic" in hp_arrays_dict:
        batch_wrappers["network_critic"] = build_hyperparam_batch(
            array=jnp.array(hp_arrays_dict["network_critic"]),
            expected_fields=MLPVectorizedHyperparams._fields,  # Assumes MLP for now TODO
            base_config_component=config.algorithm.network.critic_network,
        )
        logger.info(
            f"Built 'network_critic' HP batch wrapper. Shape: {batch_wrappers['network_critic'].shape}"
        )

    # TODO include network batches into sorting logic!
    # --- Sorting and State Preparation ---
    algo_batch_wrapper_for_sorting = batch_wrappers["algo"]
    sorted_algo_batch_wrapper, _, sort_indices = sort_hyperparams_by_steps(
        algo_batch_wrapper_for_sorting, save_dir=config.logger.base_exp_path
    )

    sorted_batch_wrappers = {"algo": sorted_algo_batch_wrapper}
    for key, wrapper in batch_wrappers.items():
        if key != "algo":
            sorted_batch_wrappers[key] = (
                wrapper.get_slice(sort_indices) if wrapper is not None else None
            )

    original_ids_array = jnp.array(config.training.hyperparam_batch_sample_ids)
    new_sample_ids = list(original_ids_array[sort_indices])
    updated_training_config = replace(config.training, hyperparam_batch_sample_ids=new_sample_ids)
    config = replace(config, training=updated_training_config)

    # --- Environment and System Setup ---
    env, eval_env = environments.make(config=config)
    config = replace(config, training=replace(config.training, num_devices=len(jax.devices())))
    config = update_config_with_env_info(config, env)

    base_key = jax.random.PRNGKey(config.training.seed)
    algo_specific_keys_tuple = algo_interface.key_setup_fn(base_key)

    normalizer_fns = normalizer_setup(
        normalize_observations=True,  # This will be controlled by the HP inside the core loop
        normalize_method=config.training.normalize_method,
        obs_spec=env.observation_spec(),
    )

    algo_setup_fns_instance = algo_interface.algo_setup_fns_factory()
    global_setup_args_for_algo = AlgorithmGlobalSetupArgs(
        env=env,
        eval_env=eval_env,
        config=config,
        normalizer_fns=normalizer_fns,
        get_eval_act_fn_callback=algo_interface.get_eval_act_fn_callback_for_algo,
        algo_specific_keys=algo_specific_keys_tuple,
    )

    # --- Phased Training Execution ---
    target_total_steps = config.training.total_timesteps
    num_evaluation_milestones = config.training.num_evaluation
    initial_num_hyperparams = sorted_algo_batch_wrapper.shape[0]
    hp_steps_per_update = (
        (sorted_algo_batch_wrapper.rollout_length * sorted_algo_batch_wrapper.total_num_envs)
        .astype(jnp.int32)
        .tolist()
    )

    sys_monitor = SystemMonitor(log_dir=str(Path(config.logger.base_exp_path)))
    sys_monitor.start()
    observers = setup_observers(
        config,
        initial_num_hyperparams,
        target_total_steps,
        num_evaluation_milestones,
        sys_monitor,
    )

    logger.info(
        f"Calling generic phased training orchestrator for {algo_interface.algorithm_name_prefix}..."
    )
    training_result = run_training_w_phaser(
        target_total_steps=target_total_steps,
        num_evaluation_milestones=num_evaluation_milestones,
        initial_num_hyperparams=initial_num_hyperparams,
        hp_steps_per_update=hp_steps_per_update,
        algo_setup_fns=algo_setup_fns_instance,
        initial_hyperparam_configs_for_algo=sorted_batch_wrappers,  # Pass the generic dict
        non_vec_hyperparams=algo_interface.non_vectorized_hyperparams_cls,
        global_args=global_setup_args_for_algo,
        observers=observers,
        include_final_master_state=False,
    )
    logger.info(f"{algo_interface.algorithm_name_prefix} Phased training completed!")

    # --- Process and Return Results ---
    sys_monitor.stop()
    return_trackers = training_result.return_trackers
    save_hyperparam_returns(
        return_trackers,
        Path(config.logger.base_exp_path) / "return_group_by_hyperparams.npz",
    )

    all_metrics_per_hp_list: list[dict[str, float]] = (
        get_metrics_for_optuna_from_hyperparam_trackers(
            return_trackers=return_trackers,
            initial_num_hyperparams=initial_num_hyperparams,
            objective_names=optuna_objective_names,
        )
    )
    logger.info(f"Metrics returned: {all_metrics_per_hp_list}")

    total_elapsed_time = time.time() - start_time
    human_readable_time = str(timedelta(seconds=total_elapsed_time)).split(".")[0]
    logger.info(
        f"Total execution time of run_experiment fn (HH:MM:SS): {str(human_readable_time)}"
    )

    timing_info = {
        "total_wall_time": total_elapsed_time,
        "total_jit_time": training_result.total_jit_time,
        "total_execution_time": training_result.total_execution_time,
    }
    results_path = Path(config.logger.base_exp_path) / "timing_info.json"
    try:
        # Determine mode based on whether it's a single run or a batch
        mode = "sequential_single" if config.training.hyperparam_batch_size <= 1 else "vectorized"
        results_to_save = {
            "mode": mode,
            "total_wall_time": timing_info["total_wall_time"],
            "jit_time": timing_info["total_jit_time"],
            "execution_time": timing_info["total_execution_time"],
            "hparam_batch_size": config.training.hyperparam_batch_size,
            "num_samples": config.training.hyperparam_batch_size,
        }
        with open(results_path, "w") as f:
            json.dump(results_to_save, f, indent=4)
        logger.info(f"Saved timing results to {results_path}")
    except Exception as e:
        logger.error(f"Failed to save timing results to {results_path}: {e}")

    return all_metrics_per_hp_list, config, timing_info
