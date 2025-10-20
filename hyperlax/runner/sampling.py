#+begin_src python
import copy
import datetime
import json
import logging
import time
from collections import defaultdict
from collections.abc import Callable
from dataclasses import replace
from pathlib import Path
from typing import TYPE_CHECKING, Any, TypeVar

from hyperlax.configs.main_base import BaseExperimentConfig
from hyperlax.configs.mapping import update_config_from_flat_dict
from hyperlax.hyperparam.base_types import HyperparamBatchGroup, flatten_tunables
from hyperlax.hyperparam.io_utils import (
    _cast_hyperparam_dict_to_expected_types,
    load_hyperparams_from_file,
)
from hyperlax.hyperparam.sampler import (
    IndependentSamples,
    SobolMatricesABOmit,
    SobolMatricesFull,
    _apply_joint_sampling_rules,
    generate_samples,
)
from hyperlax.network.parametric_torso import ACTIVATION_FN_TO_IDX
from hyperlax.runner.base_types import AlgoSpecificExperimentConfigContainer
from hyperlax.runner.batch_utils import (
    find_sample_id_key,
    group_samples_into_batches,
    log_batch_groups,
    log_hyperparam_sample_preview,
    slice_batches,
    sort_batch_groups_by_memory_impact,
)
from hyperlax.runner.execution import (
    TimingResult,
    run_sequential_experiments,
    run_single_experiment,
)
from hyperlax.runner.fs_utils import (
    is_experiment_complete,
    mark_experiment_complete,
)
import pandas as pd

if TYPE_CHECKING:
    from hyperlax.cli import SamplingSweepConfig

ExpCfgT = TypeVar("ExpCfgT", bound="BaseExperimentConfig")
logger = logging.getLogger(__name__)


def _identify_structural_hyperparams(config: BaseExperimentConfig) -> list[str]:
    """
    Identifies hyperparameters that define the computation structure,
    based on them being integer-valued Tunables.
    """
    if not hasattr(config, "algorithm"):
        return []
    flat_tunables = flatten_tunables(config.algorithm)
    structural_keys = [
        f"algorithm.{path}"
        for path, spec in flat_tunables.items()
        if spec.expected_type is int and not path.endswith(".sample_id") and spec.is_vectorized
    ]
    logger.debug(
        f"Identified structural (integer) hyperparameter keys for grouping: {structural_keys}"
    )
    return structural_keys


def _get_default_values_from_config(config: BaseExperimentConfig) -> dict[str, Any]:
    """Extracts all default values from Tunable fields in a config."""
    flat_map = flatten_tunables(config)
    return {path: spec.value for path, spec in flat_map.items()}


def _get_ordered_vectorized_keys(config_component: Any, prefix: str) -> list[str]:
    """
    Extracts all keys for vectorized Tunable fields from a config component,
    prepending a prefix to match the full path.
    """
    flat_map = flatten_tunables(config_component)
    # The keys from flatten_tunables are relative to the component, so we prepend the prefix.
    return [f"{prefix}.{path}" for path, spec in flat_map.items() if spec.is_vectorized]


def _build_vec_array(
    batch: HyperparamBatchGroup,
    vectorized_keys: list[str],
    vectorized_defaults: dict[str, Any],
) -> list[list[Any]]:
    """Create vectorized parameter array, converting activation strings to indices."""
    vec_array = []
    for i, vec_dict in enumerate(batch.vec_batches):
        row = []
        for key in vectorized_keys:
            # key is now the full path, e.g., 'algorithm.hyperparam.actor_lr'
            if key.endswith(".sample_id"):
                value = batch.sample_ids[i]
            else:
                value = vec_dict.get(key)
                if value is None:
                    value = batch.non_vec_values.get(key)
                if value is None:
                    value = vectorized_defaults.get(key)
                if value is None:
                    raise ValueError(f"No value found for vectorized parameter {key}")

            if key.endswith(".activation"):
                str_value = str(value)
                index_value = ACTIVATION_FN_TO_IDX.get(str_value)
                if index_value is None:
                    raise ValueError(f"Unknown activation name '{str_value}' for key '{key}'")
                value = index_value
            row.append(value)
        vec_array.append(row)
    return vec_array


def _build_non_vec_dict(batch: HyperparamBatchGroup) -> dict[str, Any]:
    """Create non-vectorized parameter dictionary with defaults."""
    non_vec = batch.default_values.copy()
    non_vec.update(batch.non_vec_values)
    return non_vec


def _build_experiment_config_for_batch(
    *, non_vec_dict: dict[str, Any], batch: HyperparamBatchGroup, base_config: ExpCfgT
) -> ExpCfgT:
    """Create typed experiment configuration from a batch group, separating HP components."""
    logger.info(f"Building experiment config for sample_ids: {batch.sample_ids}")

    final_config = update_config_from_flat_dict(base_config, non_vec_dict, root_path_prefix="")
    all_defaults = _get_default_values_from_config(base_config)
    vec_arrays = {}

    # 1. Algorithm Hyperparameters
    vec_keys_algo = _get_ordered_vectorized_keys(
        base_config.algorithm.hyperparam, "algorithm.hyperparam"
    )
    vec_defaults_algo = {k: v for k, v in all_defaults.items() if k in vec_keys_algo}
    vec_arrays["algo"] = _build_vec_array(batch, vec_keys_algo, vec_defaults_algo)

    # 2. Network Architectural Hyperparameters - Actor
    if hasattr(base_config.algorithm.network, "actor_network"):
        vec_keys_actor = _get_ordered_vectorized_keys(
            base_config.algorithm.network.actor_network,
            "algorithm.network.actor_network",
        )
        if vec_keys_actor:
            vec_defaults_actor = {k: v for k, v in all_defaults.items() if k in vec_keys_actor}
            vec_arrays["network_actor"] = _build_vec_array(
                batch, vec_keys_actor, vec_defaults_actor
            )
            logger.debug(f"Built actor network HP array with keys: {vec_keys_actor}")

    # 3. Network Architectural Hyperparameters - Critic
    if hasattr(base_config.algorithm.network, "critic_network"):
        vec_keys_critic = _get_ordered_vectorized_keys(
            base_config.algorithm.network.critic_network,
            "algorithm.network.critic_network",
        )
        if vec_keys_critic:
            vec_defaults_critic = {k: v for k, v in all_defaults.items() if k in vec_keys_critic}
            vec_arrays["network_critic"] = _build_vec_array(
                batch, vec_keys_critic, vec_defaults_critic
            )
            logger.debug(f"Built critic network HP array with keys: {vec_keys_critic}")

    modified_training_config = replace(
        final_config.training,
        hyperparam_batch_enabled=True,
        hyperparam_batch_size=len(batch.sample_ids),
        hyperparam_batch_samples=vec_arrays,
        hyperparam_batch_sample_ids=batch.sample_ids,
    )
    final_config = replace(final_config, training=modified_training_config)

    algo_name = final_config.algorithm.__class__.__name__.replace("Config", "")
    modified_experiment_tags = f"{algo_name},{datetime.datetime.now().strftime('%Y%m%d-%H%M%S')}"
    final_config = replace(final_config, experiment_tags=modified_experiment_tags)

    return final_config


def _setup_experiment_configs_for_batched(
    batches: list[HyperparamBatchGroup],
    base_config: ExpCfgT,
) -> list[ExpCfgT]:
    """Create experiment configurations from batch groups."""
    logger.info(f"--- Setting up experiment configs for {len(batches)} batch groups ---")
    if not batches:
        raise ValueError("Batches list cannot be empty")

    experiment_configs = []
    for i, batch_group in enumerate(batches):
        logger.info(f"Processing batch group {i + 1}/{len(batches)}")
        non_vec_dict = _build_non_vec_dict(batch_group)
        exp_config = _build_experiment_config_for_batch(
            non_vec_dict=non_vec_dict,
            batch=batch_group,
            base_config=base_config,
        )
        experiment_configs.append(exp_config)

    logger.info(f"Created {len(experiment_configs)} experiment configurations")
    return experiment_configs


def _run_sliced_experiments_timed(
    batch_groups: list[HyperparamBatchGroup],
    max_batch_size: int,
    min_batch_size: int | None,
    base_config: ExpCfgT,
    main_fn: Callable,
    args: "SamplingSweepConfig",
) -> tuple[TimingResult, dict[int, dict[str, Any]]]:
    """Run experiments with sliced batch groups and timing."""
    logger.info(
        f"\nSlicing batch groups with max_batch_size={max_batch_size}, min_batch_size={min_batch_size}"
    )
    sliced_groups = slice_batches(batch_groups, max_batch_size, min_batch_size)
    logger.info(f"Created {len(sliced_groups)} sliced groups")
    log_batch_groups(sliced_groups)

    logger.info("\nCreating experiment configurations...")
    configs = _setup_experiment_configs_for_batched(
        batches=sliced_groups,
        base_config=base_config,
    )
    for idx, config in enumerate(configs):
        timestamp = datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
        new_tags = (
            f"{config.algorithm._target_},"
            f"slice_{idx + 1}_of_{len(configs)},"
            f"max_batch_{max_batch_size},"
            f"{timestamp}"
        )
        batch_dir = Path(config.logger.base_exp_path) / f"batch_{idx:05d}"
        new_logger_config = replace(config.logger, base_exp_path=str(batch_dir))
        configs[idx] = replace(config, logger=new_logger_config, experiment_tags=new_tags)

        # Log and save this specific slice's hyperparameter group to its run directory
        log_batch_groups([sliced_groups[idx]], save_dir=str(batch_dir))

    timing_result, all_metrics = run_sequential_experiments(configs, main_fn, args)
    return timing_result, all_metrics


def _process_single_sample_set(
    sample_name: str,
    samples: dict[str, list[Any]],
    base_config: ExpCfgT,
    args: "SamplingSweepConfig",
    main_fn: Callable,
) -> tuple[TimingResult | None, dict[int, dict[str, Any]]]:
    """Helper to process a single dictionary of hyperparameter samples (e.g., matrix A or B)."""
    logger.info(f"\n--- Processing Sample Set: {sample_name} ---")

    try:
        sample_id_key = find_sample_id_key(base_config)
    except ValueError as e:
        logger.error(
            f"Cannot process sample set '{sample_name}': {e}. This may happen if the config doesn't have a 'Tunable' field for 'sample_id'."
        )
        return None, {}

    num_samples = len(samples.get(sample_id_key, []))
    if num_samples == 0:
        logger.info(
            f"Skipping either empty sample set or could not infer sample_id_key: {sample_name}"
        )
        return None, {}

    samples = _cast_hyperparam_dict_to_expected_types(samples, base_config)
    logger.debug("Samples after type cast:")
    log_hyperparam_sample_preview(samples)

    flat_tunables = flatten_tunables(base_config)
    vectorized_keys = {path for path, spec in flat_tunables.items() if spec.is_vectorized}
    non_vectorized_keys = {path for path, spec in flat_tunables.items() if not spec.is_vectorized}
    all_defaults = _get_default_values_from_config(base_config)

    batch_groups_to_run: list[HyperparamBatchGroup] = []
    if args.group_by_structural_hparams:
        logger.info("\nGrouping samples by structural hyperparameters...")
        samples_df = pd.DataFrame.from_dict(samples)
        structural_keys = _identify_structural_hyperparams(base_config)
        existing_structural_keys = [k for k in structural_keys if k in samples_df.columns]

        groups_to_process: list[dict[str, list[Any]]]
        if not existing_structural_keys:
            logger.warning("No structural keys found, falling back to default grouping.")
            groups_to_process = [samples]
        else:
            logger.info(f"Partitioning samples into groups based on: {existing_structural_keys}")
            df_groups = samples_df.groupby(existing_structural_keys)
            groups_to_process = [group_df.to_dict(orient="list") for _, group_df in df_groups]
            logger.info(f"Created {len(groups_to_process)} homogeneous groups.")

        for group_samples_dict in groups_to_process:
            # This inner grouping handles any other non-vectorized params within the structural group.
            batch_groups = group_samples_into_batches(
                group_samples_dict,
                vectorized_keys,
                non_vectorized_keys,
                all_defaults,
                sample_id_key=sample_id_key,
            )
            batch_groups_to_run.extend(batch_groups)
    else:
        logger.info("\nGrouping samples...")
        batch_groups_to_run = group_samples_into_batches(
            samples,
            vectorized_keys,
            non_vectorized_keys,
            all_defaults,
            sample_id_key=sample_id_key,
        )

    log_batch_groups(batch_groups_to_run)
    logger.info("\nSorting all group samples...")
    sorted_groups = sort_batch_groups_by_memory_impact(batch_groups_to_run)
    log_batch_groups(sorted_groups)

    timing_result, all_metrics = _run_sliced_experiments_timed(
        batch_groups=sorted_groups,
        max_batch_size=args.hparam_batch_size,
        min_batch_size=args.min_hparam_batch_size,
        base_config=base_config,
        main_fn=main_fn,
        args=args,
    )

    logger.info(f"\n--- Timing Summary for {sample_name} ---")
    logger.info(str(timing_result))
    logger.info("-" * 40)
    return timing_result, all_metrics


def process_batched_sample_set(
    args: "SamplingSweepConfig",
    exp_config_container: AlgoSpecificExperimentConfigContainer,
    predefined_samples: dict[str, list[Any]] | None = None,
) -> tuple[TimingResult | None, dict[int, dict[str, Any]]]:
    """
    Main entry point for batched experiment runs. Handles different experiment types.
    """
    base_config = exp_config_container.experiment_config
    main_fn = exp_config_container.algo_main_fn
    output_dir = Path(args.output_dir)

    samples_to_process = []
    if predefined_samples:
        config = copy.deepcopy(base_config)
        new_logger_config = replace(
            config.logger, base_exp_path=str(output_dir / "predefined_samples")
        )
        config = replace(config, logger=new_logger_config)
        samples_to_process.append(("predefined", predefined_samples, config))
    elif args.load_hyperparams:
        flat_tunables = flatten_tunables(base_config)
        samples = load_hyperparams_from_file(
            args.load_hyperparams, list(flat_tunables.keys()), base_config
        )
        samples_to_process.append(("Loaded", samples, base_config))
    else:
        sampling_result = generate_samples(args, exp_config_container)
        if isinstance(sampling_result, IndependentSamples):
            config = copy.deepcopy(base_config)
            new_logger_config = replace(config.logger, base_exp_path=str(output_dir / "samples"))
            config = replace(config, logger=new_logger_config)
            samples_to_process.append(("Independent", sampling_result.unn, config))
        elif isinstance(sampling_result, (SobolMatricesFull, SobolMatricesABOmit)):
            config_A = copy.deepcopy(base_config)
            new_logger_config_A = replace(
                config_A.logger, base_exp_path=str(output_dir / "sample_A")
            )
            config_A = replace(config_A, logger=new_logger_config_A)
            samples_to_process.append(("Sobol_A", sampling_result.A_unn, config_A))

            config_B = copy.deepcopy(base_config)
            new_logger_config_B = replace(
                config_B.logger, base_exp_path=str(output_dir / "sample_B")
            )
            config_B = replace(config_B, logger=new_logger_config_B)
            samples_to_process.append(("Sobol_B", sampling_result.B_unn, config_B))

            if isinstance(sampling_result, SobolMatricesFull):
                for param_name, ab_dict in sampling_result.AB_unn_list:
                    config_AB = copy.deepcopy(base_config)
                    new_logger_config_AB = replace(
                        config_AB.logger,
                        base_exp_path=str(output_dir / "sample_AB" / param_name),
                    )
                    config_AB = replace(config_AB, logger=new_logger_config_AB)
                    samples_to_process.append((f"Sobol_AB_{param_name}", ab_dict, config_AB))

    if not samples_to_process:
        logger.warning("No samples were generated or loaded. Nothing to run.")
        return None, {}

    all_metrics = {}
    aggregated_timing_result = None

    # Aggregate timing across all sample sets for batched runs
    total_batched_wall_time = 0.0
    total_batched_jit_time = 0.0
    total_batched_exec_time = 0.0
    total_batched_samples = 0

    for name, samples, config in samples_to_process:
        # NOTE: Joint sampling rules are applied inside `generate_samples` before saving.
        # Predefined/loaded samples are handled here.
        if predefined_samples or args.load_hyperparams:
            processed_samples = _apply_joint_sampling_rules(samples, exp_config_container)
        else:
            processed_samples = samples  # Already processed inside generate_samples

        timing_result, metrics = _process_single_sample_set(
            sample_name=name,
            samples=processed_samples,
            base_config=config,
            args=args,
            main_fn=main_fn,
        )
        all_metrics.update(metrics)

        # Aggregate timing results across all sample sets
        if timing_result:
            aggregated_timing_result = timing_result
            total_batched_wall_time += timing_result.total_time
            total_batched_jit_time += timing_result.total_jit_time
            total_batched_exec_time += timing_result.total_execution_time
            total_batched_samples += timing_result.num_experiments

    # Save the aggregated timing info for the entire batched run at the top level
    results_path = output_dir / "timing_info.json"
    try:
        results_to_save = {
            "mode": "batched",
            "total_wall_time": total_batched_wall_time,
            "jit_time": total_batched_jit_time,
            "execution_time": total_batched_exec_time,
            "num_samples": total_batched_samples,
            "hparam_batch_size": args.hparam_batch_size,
        }
        with open(results_path, "w") as f:
            json.dump(results_to_save, f, indent=4)
        logger.info(f"Saved aggregated batched timing results to {results_path}")
    except Exception as e:
        logger.error(f"Failed to save aggregated batched timing results: {e}")

    return aggregated_timing_result, all_metrics

def _run_sequential_for_single_set(
    sample_name: str,
    samples: dict[str, list[Any]],
    base_config: ExpCfgT,
    args: "SamplingSweepConfig",
    main_fn: Callable,
) -> dict:
    """Helper to run a single set of samples sequentially (e.g., Matrix A) and return timing info."""
    if not samples or not any(len(v) > 0 for v in samples.values()):
        logger.info(f"No samples in '{sample_name}' set to run sequentially.")
        return {
            "total_wall_time": 0,
            "jit_time": 0,
            "execution_time": 0,
            "num_samples": 0,
        }

    try:
        sample_id_key = find_sample_id_key(base_config)
    except ValueError as e:
        logger.error(f"Cannot run sequential set: {e}")
        return {
            "total_wall_time": 0,
            "jit_time": 0,
            "execution_time": 0,
            "num_samples": 0,
        }

    num_samples = len(next(iter(samples.values())))
    all_keys = list(samples.keys())

    set_output_dir = Path(base_config.logger.base_exp_path)
    set_output_dir.mkdir(parents=True, exist_ok=True)

    logger.info(f"\n--- Running Sequential Sweep for Set: {sample_name} in {set_output_dir} ---")

    total_wall_time, total_jit_time, total_exec_time = 0.0, 0.0, 0.0

    for i in range(num_samples):
        sample_start_time = time.perf_counter()

        sample_dict = {key: samples[key][i] for key in all_keys}
        sample_id = int(sample_dict.get(sample_id_key, i))

        run_dir = set_output_dir / f"run_{sample_id:04d}"
        if args.resume and is_experiment_complete(run_dir):
            logger.info(
                f"Skipping run {sample_id} in set '{sample_name}' as it is already complete."
            )
            continue

        run_dir.mkdir(parents=True, exist_ok=True)

        # 1. Update the base config with the flat dictionary of hyperparams for this sample
        config_for_run = update_config_from_flat_dict(base_config, sample_dict)

        # 2. Since this is a single run, hyperparam_batch should be disabled in the config.
        # The main_fn will receive a config object as if it's a non-batched experiment.

        hp_batch_samples = defaultdict(list)
        flat_tunables_run = flatten_tunables(config_for_run)

        vec_keys_algo = _get_ordered_vectorized_keys(
            config_for_run.algorithm.hyperparam, "algorithm.hyperparam"
        )
        vec_keys_actor = (
            _get_ordered_vectorized_keys(
                config_for_run.algorithm.network.actor_network,
                "algorithm.network.actor_network",
            )
            if hasattr(config_for_run.algorithm.network, "actor_network")
            else []
        )
        vec_keys_critic = (
            _get_ordered_vectorized_keys(
                config_for_run.algorithm.network.critic_network,
                "algorithm.network.critic_network",
            )
            if hasattr(config_for_run.algorithm.network, "critic_network")
            else []
        )

        algo_row = [
            flat_tunables_run[key].value for key in vec_keys_algo if not key.endswith("sample_id")
        ]
        algo_row.append(sample_id)
        hp_batch_samples["algo"] = [algo_row]

        if vec_keys_actor:
            actor_row = [
                (
                    ACTIVATION_FN_TO_IDX.get(flat_tunables_run[k].value)
                    if k.endswith(".activation")
                    else flat_tunables_run[k].value
                )
                for k in vec_keys_actor
            ]
            hp_batch_samples["network_actor"] = [actor_row]

        if vec_keys_critic:
            critic_row = [
                (
                    ACTIVATION_FN_TO_IDX.get(flat_tunables_run[k].value)
                    if k.endswith(".activation")
                    else flat_tunables_run[k].value
                )
                for k in vec_keys_critic
            ]
            hp_batch_samples["network_critic"] = [critic_row]

        training_cfg = replace(
            config_for_run.training,
            hyperparam_batch_enabled=True,
            hyperparam_batch_size=1,  # conceptually 1, but disabled
            hyperparam_batch_samples=dict(hp_batch_samples),  # No batch samples needed
            hyperparam_batch_sample_ids=[sample_id],
        )
        logger_cfg = replace(
            config_for_run.logger, base_exp_path=str(run_dir), enable_timing_logs=True
        )
        final_run_config = replace(config_for_run, training=training_cfg, logger=logger_cfg)

        logger.info(
            f"--- Running Sequential Sample {i + 1}/{num_samples} (Set: {sample_name}, ID: {sample_id}) ---"
        )
        success, _, timing_info = run_single_experiment(final_run_config, main_fn)

        if success:
            mark_experiment_complete(run_dir)
            total_jit_time += timing_info.get("total_jit_time", 0.0)
            total_exec_time += timing_info.get("total_execution_time", 0.0)

        total_wall_time += time.perf_counter() - sample_start_time

    return {
        "total_wall_time": total_wall_time,
        "jit_time": total_jit_time,
        "execution_time": total_exec_time,
        "num_samples": num_samples,
    }


def process_sequential_sample_set(
    args: "SamplingSweepConfig",
    exp_config_container: AlgoSpecificExperimentConfigContainer,
) -> None:
    """Main entry point for sequential experiment runs, supporting all experiment types."""
    base_config = exp_config_container.experiment_config
    main_fn = exp_config_container.algo_main_fn
    output_dir = Path(args.output_dir)

    samples_to_process = []
    if args.load_hyperparams:
        flat_tunables = flatten_tunables(base_config)
        samples = load_hyperparams_from_file(
            args.load_hyperparams, list(flat_tunables.keys()), base_config
        )
        samples_to_process.append(("Loaded", samples, base_config))
    else:
        sampling_result = generate_samples(args, exp_config_container)
        if isinstance(sampling_result, IndependentSamples):
            config = copy.deepcopy(base_config)
            new_logger_config = replace(config.logger, base_exp_path=str(output_dir / "samples"))
            config = replace(config, logger=new_logger_config)
            samples_to_process.append(("Independent", sampling_result.unn, config))
        elif isinstance(sampling_result, (SobolMatricesFull, SobolMatricesABOmit)):
            config_A = copy.deepcopy(base_config)
            new_logger_config_A = replace(
                config_A.logger, base_exp_path=str(output_dir / "sample_A")
            )
            config_A = replace(config_A, logger=new_logger_config_A)
            samples_to_process.append(("Sobol_A", sampling_result.A_unn, config_A))

            config_B = copy.deepcopy(base_config)
            new_logger_config_B = replace(
                config_B.logger, base_exp_path=str(output_dir / "sample_B")
            )
            config_B = replace(config_B, logger=new_logger_config_B)
            samples_to_process.append(("Sobol_B", sampling_result.B_unn, config_B))

            if isinstance(sampling_result, SobolMatricesFull):
                for param_name, ab_dict in sampling_result.AB_unn_list:
                    config_AB = copy.deepcopy(base_config)
                    new_logger_config_AB = replace(
                        config_AB.logger,
                        base_exp_path=str(output_dir / "sample_AB" / param_name),
                    )
                    config_AB = replace(config_AB, logger=new_logger_config_AB)
                    samples_to_process.append((f"Sobol_AB_{param_name}", ab_dict, config_AB))

    if not samples_to_process:
        logger.warning("No samples were generated or loaded for sequential run.")
        return

    # Aggregate timing results for the entire sequential sweep
    total_sequential_wall_time = 0.0
    total_sequential_jit_time = 0.0
    total_sequential_exec_time = 0.0
    total_sequential_samples = 0

    for name, samples, config in samples_to_process:
        # NOTE: Joint sampling rules are applied inside `generate_samples` before saving.
        # Predefined/loaded samples are handled here.
        if args.load_hyperparams:
            processed_samples = _apply_joint_sampling_rules(samples, exp_config_container)
        else:
            processed_samples = samples  # Already processed inside generate_samples

        set_timing_info = _run_sequential_for_single_set(
            sample_name=name,
            samples=processed_samples,
            base_config=config,
            args=args,
            main_fn=main_fn,
        )
        total_sequential_wall_time += set_timing_info["total_wall_time"]
        total_sequential_jit_time += set_timing_info["jit_time"]
        total_sequential_exec_time += set_timing_info["execution_time"]
        total_sequential_samples += set_timing_info["num_samples"]

    # Save the aggregated timing info for the entire sequential run
    results_path = output_dir / "timing_info.json"
    try:
        results_to_save = {
            "mode": "sequential",
            "total_wall_time": total_sequential_wall_time,
            "jit_time": total_sequential_jit_time,
            "execution_time": total_sequential_exec_time,
            "num_samples": total_sequential_samples,
            "hparam_batch_size": 1,  # By definition
        }
        with open(results_path, "w") as f:
            json.dump(results_to_save, f, indent=4)
        logger.info(f"Saved aggregated sequential timing results to {results_path}")
    except Exception as e:
        logger.error(f"Failed to save aggregated sequential timing results: {e}")
