import logging
import os
import time
from collections.abc import Callable
from contextlib import contextmanager
from typing import TYPE_CHECKING, Any, NamedTuple, TypeVar

import jax.numpy as jnp
import numpy as np

from hyperlax.configs.main_base import BaseExperimentConfig
from hyperlax.configs.serialization import save_experiment_config

# from hyperlax.runner.base_types import ArgsConfig
from hyperlax.runner.fs_utils import is_experiment_complete, mark_experiment_complete

if TYPE_CHECKING:
    from hyperlax.cli import SamplingSweepConfig


ExpCfgT = TypeVar("ExpCfgT", bound="BaseExperimentConfig")
logger = logging.getLogger(__name__)


class TimingResult(NamedTuple):
    """Container for experiment timing results."""

    total_time: float
    experiment_times: list[float]
    avg_time: float
    max_time: float
    min_time: float
    execution_mode: str
    batch_size_per_group: list[int]
    num_experiments: int
    successful_experiments: int = 0
    total_jit_time: float = 0.0
    total_execution_time: float = 0.0

    def __str__(self) -> str:
        return (
            f"Execution Mode: {self.execution_mode}\n"
            f"Batch Sizes per Group: {self.batch_size_per_group}\n"
            f"Number of Experiments (Sliced Groups): {self.num_experiments}\n"
            f"Total Time: {self.total_time:.2f}s\n"
            f"Average Time per Experiment: {self.avg_time:.2f}s\n"
            f"Max Time: {self.max_time:.2f}s\n"
            f"Min Time: {self.min_time:.2f}s\n"
            f"Total JIT Time (sum over groups): {self.total_jit_time:.2f}s\n"
            f"Total Pure Execution Time (sum over groups): {self.total_execution_time:.2f}s"
        )


@contextmanager
def timer():
    """Context manager for timing code blocks."""
    start = time.perf_counter()
    yield lambda: time.perf_counter() - start


def run_single_experiment(
    config: ExpCfgT, main_fn: Callable
) -> tuple[bool, dict[int, dict[str, Any]], dict[str, float]]:
    """Run a single experiment with the given configuration."""
    os.makedirs(config.logger.base_exp_path, exist_ok=True)
    logger.info(f"Saving experiment config to {config.logger.base_exp_path}/config.yaml")
    save_experiment_config(config, os.path.join(config.logger.base_exp_path, "config.yaml"))
    metrics_for_this_run_batch: dict[int, dict[str, Any]] = {}
    success = False
    timing_info = {}
    try:
        all_metrics_per_hp_list: list[dict[str, float]]
        config_final: BaseExperimentConfig

        all_metrics_per_hp_list, config_final, timing_info = main_fn(config)
        success = True

        sample_ids_raw = config_final.training.hyperparam_batch_sample_ids
        processed_sample_ids: list[int] = []

        for sid_val in sample_ids_raw:
            if isinstance(sid_val, (jnp.ndarray, np.ndarray)):
                if sid_val.ndim == 0:
                    processed_sample_ids.append(int(sid_val.item()))
                elif sid_val.size == 1:
                    processed_sample_ids.append(int(sid_val.reshape(-1)[0].item()))
                else:
                    raise ValueError(
                        f"Sample ID array '{sid_val}' has multiple elements or is not scalar."
                    )
            elif isinstance(sid_val, (int, float, np.integer, np.floating)):
                processed_sample_ids.append(int(sid_val))
            else:
                try:
                    processed_sample_ids.append(int(sid_val))
                except (TypeError, ValueError) as e_direct_conv:
                    raise ValueError(
                        f"Sample ID '{sid_val}' (type: {type(sid_val)}) is not convertible to int."
                    ) from e_direct_conv

        if len(processed_sample_ids) != len(all_metrics_per_hp_list):
            logger.error(
                f"ERROR: Mismatch in length between processed sample IDs ({len(processed_sample_ids)}) "
                f"and returned metrics list ({len(all_metrics_per_hp_list)}). Critical data alignment issue."
            )
            success = False
        else:
            for i, py_sample_id in enumerate(processed_sample_ids):
                current_hp_metrics_dict = {
                    k: (float(v) if isinstance(v, (jnp.ndarray, np.ndarray, np.number)) else v)
                    for k, v in all_metrics_per_hp_list[i].items()
                }
                metrics_for_this_run_batch[py_sample_id] = current_hp_metrics_dict
                logger.info(
                    f"  Metrics for sample_id {py_sample_id}: {metrics_for_this_run_batch[py_sample_id]}"
                )

    except Exception:
        logger.exception(f"Error in experiment ({config.logger.base_exp_path})")
        success = False
    return success, metrics_for_this_run_batch, timing_info


def run_sequential_experiments(
    configs: list[ExpCfgT], main_fn: Callable, args: "SamplingSweepConfig"
) -> tuple[TimingResult, dict[int, dict[str, Any]]]:
    """Runs experiments sequentially with timing."""
    logger.info("\nRunning batch of hyperparams...")
    experiment_times = []
    batch_sizes = [cfg.training.hyperparam_batch_size for cfg in configs]
    successful_experiments = 0
    all_metrics = {}
    total_jit_time_agg = 0.0
    total_exec_time_agg = 0.0

    with timer() as get_total_time:
        for idx, config in enumerate(configs):
            if args.resume and is_experiment_complete(config.logger.base_exp_path):
                logger.info(f"\nSkipping batch {idx + 1}/{len(configs)} - already completed")
                continue

            logger.info(f"\nExecuting experiment {idx + 1}/{len(configs)}")
            logger.info(f"Configuration Tags: {config.experiment_tags}")
            logger.info(f"Hyperparam Batch Size: {config.training.hyperparam_batch_size}")

            with timer() as get_exp_time:
                success, metrics_for_run, timing_info = run_single_experiment(config, main_fn)
                if success:
                    mark_experiment_complete(config.logger.base_exp_path)
                    successful_experiments += 1
                    experiment_times.append(get_exp_time())
                    all_metrics.update(metrics_for_run)
                    total_jit_time_agg += timing_info.get("total_jit_time", 0.0)
                    total_exec_time_agg += timing_info.get("total_execution_time", 0.0)
                else:
                    logger.error(f"Experiment {idx + 1} failed, not marking as complete.")

    total_time = get_total_time()
    timing_result = TimingResult(
        total_time=total_time,
        experiment_times=experiment_times,
        avg_time=(sum(experiment_times) / len(experiment_times) if experiment_times else 0),
        max_time=max(experiment_times) if experiment_times else 0,
        min_time=min(experiment_times) if experiment_times else 0,
        execution_mode="sequential",
        batch_size_per_group=batch_sizes,
        num_experiments=len(configs),
        successful_experiments=successful_experiments,
        total_jit_time=total_jit_time_agg,
        total_execution_time=total_exec_time_agg,
    )
    return timing_result, all_metrics
