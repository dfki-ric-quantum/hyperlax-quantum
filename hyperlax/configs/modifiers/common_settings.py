import dataclasses
import logging

from hyperlax.configs.main_base import BaseExperimentConfig
from hyperlax.hyperparam.tunable import Tunable

logger = logging.getLogger(__name__)


def apply_quick_test_settings(config: BaseExperimentConfig) -> BaseExperimentConfig:
    """
    Returns a NEW config object with settings for a quick test run.
    This function is now purely functional and correctly modifies Tunable objects.
    """
    # Create a new, modified training config
    new_training_config = dataclasses.replace(
        config.training,
        total_timesteps=40,
        num_evaluation=4,
        num_eval_episodes=4,
        num_agents_slash_seeds=2,
    )

    # Create a new, modified logger config
    new_logger_config = dataclasses.replace(
        config.logger,
        level="INFO",
        enable_hyperparam_progress_bar=False,
        enable_jax_debug_prints=False,  # Only True if intentional!
    )

    # Start with a copy of the existing hyperparam config
    new_hyperparam_config = config.algorithm.hyperparam

    # Define the quick-test values
    quick_hp_values = {
        "total_num_envs": 4,
        "total_buffer_size": 100,
        "total_batch_size": 2,
        "warmup_rollout_length": 2,
        "rollout_length": 1,
        "epochs": 1,
        "num_minibatches": 2,
    }

    # Iterate through the fields of the hyperparam config
    hp_replacements = {}
    for field in dataclasses.fields(new_hyperparam_config):
        if field.name in quick_hp_values and isinstance(
            getattr(new_hyperparam_config, field.name), Tunable
        ):
            # Get the existing Tunable object
            original_tunable = getattr(new_hyperparam_config, field.name)
            # Create a new Tunable object with the value replaced
            new_tunable = dataclasses.replace(original_tunable, value=quick_hp_values[field.name])
            hp_replacements[field.name] = new_tunable

    # Apply the replacements to the hyperparam config
    if hp_replacements:
        new_hyperparam_config = dataclasses.replace(new_hyperparam_config, **hp_replacements)

    # Create the new algorithm component config with the updated hyperparam object
    new_algorithm_component = dataclasses.replace(
        config.algorithm, hyperparam=new_hyperparam_config
    )

    # Assemble the final, new experiment config
    final_config = dataclasses.replace(
        config,
        algorithm=new_algorithm_component,
        training=new_training_config,
        logger=new_logger_config,
    )

    logger.info("Applied quick test run settings!")

    return final_config


def apply_long_run_settings(config: BaseExperimentConfig) -> BaseExperimentConfig:
    new_training_config = dataclasses.replace(
        config.training,
        total_timesteps=int(5e6),
        num_evaluation=40,
        num_eval_episodes=32,
        num_agents_slash_seeds=4,
    )
    new_logger_config = dataclasses.replace(
        config.logger, level="ERROR", enable_hyperparam_progress_bar=False
    )
    mod_config = dataclasses.replace(
        config, training=new_training_config, logger=new_logger_config
    )
    logger.info("Applied long run settings!")
    return mod_config


def apply_runtime_benchmark_settings(config: BaseExperimentConfig, pqc_mode: bool = False) -> BaseExperimentConfig:
    """
    Returns a NEW config object with settings optimized for benchmarking runtime.
    Disables all non-essential logging, metrics, and evaluations to isolate core training logic.
    """
    # Create a new, modified training config
    new_training_config = dataclasses.replace(
        config.training,
        num_evaluation=0,  # Disable evaluations entirely
        num_eval_episodes=0,
        total_timesteps=int(2e4) if pqc_mode else int(1e6),
    )

    # Create a new, modified logger config
    new_logger_config = dataclasses.replace(
        config.logger,
        level="CRITICAL",  # Suppress all logs except critical errors
        save_console_to_file=False,
        aggregate_metrics=False,
        enable_hyperparam_progress_bar=False,  # Disables SystemMonitor
        enable_jax_debug_prints=False,
        enable_timing_logs=False,
        enable_gpu_memory_logging=False,
        checkpointing_enabled=False,
        enable_summarize_layout=False,
    )

    # Assemble the final, new experiment config. No changes to algorithm HPs needed.
    final_config = dataclasses.replace(
        config,
        training=new_training_config,
        logger=new_logger_config,
    )

    logger.info("Applied runtime benchmark settings: evaluations and extensive logging disabled.")

    return final_config

def apply_runtime_benchmark_settings_given_timesteps(config: BaseExperimentConfig, timesteps: bool = False) -> BaseExperimentConfig:
    """
    Returns a NEW config object with settings optimized for benchmarking runtime.
    Disables all non-essential logging, metrics, and evaluations to isolate core training logic.
    """
    # Create a new, modified training config
    new_training_config = dataclasses.replace(
        config.training,
        num_evaluation=0,  # Disable evaluations entirely
        num_eval_episodes=0,
        total_timesteps=timesteps,
    )

    # Create a new, modified logger config
    new_logger_config = dataclasses.replace(
        config.logger,
        level="CRITICAL",  # Suppress all logs except critical errors
        save_console_to_file=False,
        aggregate_metrics=False,
        enable_hyperparam_progress_bar=False,  # Disables SystemMonitor
        enable_jax_debug_prints=False,
        enable_timing_logs=False,
        enable_gpu_memory_logging=False,
        checkpointing_enabled=False,
        enable_summarize_layout=False,
    )

    # Assemble the final, new experiment config. No changes to algorithm HPs needed.
    final_config = dataclasses.replace(
        config,
        training=new_training_config,
        logger=new_logger_config,
    )

    logger.info("Applied runtime benchmark settings: evaluations and extensive logging disabled.")

    return final_config
