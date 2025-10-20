import logging
from collections.abc import Callable
from dataclasses import replace
from pathlib import Path
from typing import Any

from hyperlax.runner.execution import run_single_experiment
from hyperlax.runner.fs_utils import mark_experiment_complete
from hyperlax.runner.sampling import _build_experiment_config

logger = logging.getLogger(__name__)


def run_sequential_sweep_over_samples(
    samples: dict[str, list[Any]],
    base_config: Any,
    main_fn: Callable,
    output_dir: str,
):
    """
    For each hyperparam sample, run a single experiment sequentially.
    Each run gets its own run_XXXX directory.
    """
    if not samples or not any(len(v) > 0 for v in samples.values()):
        logger.info("No samples provided for sequential sweep.")
        return

    num_individual_samples = len(next(iter(samples.values())))

    individual_sample_dicts = []
    keys_in_samples = list(samples.keys())
    for i in range(num_individual_samples):
        current_sample_dict = {key: samples[key][i] for key in keys_in_samples}
        individual_sample_dicts.append(current_sample_dict)

    for idx, sample_as_non_vec_dict in enumerate(individual_sample_dicts):
        run_dir = Path(output_dir) / f"run_{idx:04d}"
        run_dir.mkdir(parents=True, exist_ok=True)

        current_run_config = _build_experiment_config(
            non_vec_dict=sample_as_non_vec_dict,
            vec_array=[],
            sample_ids=[],
            base_config=base_config,
        )

        new_logger_config = replace(current_run_config.logger, base_exp_path=str(run_dir))
        new_training_config = replace(
            current_run_config.training,
            hyperparam_batch_enabled=False,
            hyperparam_batch_size=0,
            hyperparam_batch_samples=[],
            hyperparam_batch_sample_ids=[],
        )
        current_run_config = replace(
            current_run_config,
            logger=new_logger_config,
            training=new_training_config,
            experiment_mode="single",
        )

        logger.info(
            f"Running sequential experiment {idx + 1}/{num_individual_samples} in {run_dir}"
        )

        success, _ = run_single_experiment(current_run_config, main_fn)
        if success:
            mark_experiment_complete(run_dir)
            logger.info(f"Completed run {idx + 1}/{num_individual_samples} in {run_dir}")
        else:
            logger.error(f"Run {idx + 1}/{num_individual_samples} failed in {run_dir}")
