import copy
import dataclasses
import json
import logging
import traceback
from collections import defaultdict
from typing import Any

import optuna
from optuna.visualization import (
    plot_contour,
    plot_optimization_history,
    plot_parallel_coordinate,
    plot_param_importances,
    plot_pareto_front,
    plot_slice,
)

from hyperlax.hyperparam.base_types import BaseDistribution
from hyperlax.hyperparam.distributions import (
    Categorical,
    LogUniform,
    UniformContinuous,
    UniformDiscrete,
)
from hyperlax.runner.base_types import AlgoSpecificExperimentConfigContainer
from hyperlax.runner.batch_utils import find_sample_id_key
from hyperlax.runner.launch_args import OptunaSweepConfig, SamplingSweepConfig
from hyperlax.runner.sampling import process_batched_sample_set

logger = logging.getLogger(__name__)


def _suggest_from_dist(trial: optuna.Trial, name: str, dist: BaseDistribution) -> Any:
    """Converts a hyperlax distribution to an Optuna suggestion."""
    if isinstance(dist, LogUniform):
        return trial.suggest_float(name, dist.domain[0], dist.domain[1], log=True)
    elif isinstance(dist, UniformContinuous):
        return trial.suggest_float(name, dist.domain[0], dist.domain[1], log=False)
    elif isinstance(dist, Categorical):
        # Optuna's suggest_categorical requires choices to be basic types.
        # We serialize complex types (like lists) to JSON strings.
        all_choices_are_optuna_basic = all(
            isinstance(v, (str, int, float, bool, type(None))) for v in dist.values
        )
        if all_choices_are_optuna_basic:
            return trial.suggest_categorical(name, dist.values)
        else:
            try:
                str_choices = [json.dumps(v) for v in dist.values]
                suggested_str_choice = trial.suggest_categorical(name, str_choices)
                return json.loads(suggested_str_choice)
            except (TypeError, json.JSONDecodeError) as e:
                trial.study.logger.error(
                    f"CRITICAL ERROR: Could not serialize/deserialize categorical choice for '{name}'. "
                    f"Values: {dist.values}. Error: {e}. This parameter cannot be suggested."
                )
                raise optuna.exceptions.TrialPruned(
                    f"Failed to suggest parameter {name} due to serialization issues."
                ) from e
    elif isinstance(dist, UniformDiscrete):
        return trial.suggest_int(name, dist.domain[0], dist.domain[1])
    else:
        raise TypeError(
            f"Unsupported distribution type for Optuna: {type(dist)} for parameter '{name}'"
        )


def run_optuna_sweep_impl(
    args: "OptunaSweepConfig", exp_info: AlgoSpecificExperimentConfigContainer
):
    """
    Core implementation for running an Optuna hyperparameter optimization study.
    This function contains the logic adapted from the original optuna_vectorized_launcher.py.
    """
    # Unpack from the experiment info container
    base_config_template = exp_info.experiment_config
    hyperparam_distributions = exp_info.hyperparam_dist_config
    algorithm_main_fn = exp_info.algo_main_fn
    output_dir = args.output_dir

    # --- 1. Setup Optuna Study ---
    logger.info(f"Starting Optuna study '{args.study_name}'...")
    logger.info(f"Objectives: {args.objective_names}, Directions: {args.objective_directions}")

    # Find the dynamic key ONCE at the beginning
    try:
        sample_id_key = find_sample_id_key(base_config_template)
    except ValueError as e:
        logger.error(
            f"FATAL: Could not find 'sample_id' Tunable in config. Optuna sweep cannot proceed. Error: {e}"
        )
        return

    # Construct the full, absolute path for the SQLite database URL
    db_path = output_dir / args.storage
    storage_url = f"sqlite:///{db_path.resolve()}"
    logger.info(f"Optuna study storage URL: {storage_url}")

    optuna_plots_dir = output_dir / "optuna_plots"

    sampler_map = {
        "tpe": optuna.samplers.TPESampler,
        "nsgaii": optuna.samplers.NSGAIISampler,
    }
    optuna_sampler = sampler_map[args.sampler](seed=args.seed)

    study = optuna.create_study(
        study_name=args.study_name,
        storage=storage_url,
        directions=args.objective_directions,
        load_if_exists=args.resume,
        sampler=optuna_sampler,
    )
    try:
        current_names = study.user_attrs.get("objective_names")
        if current_names != args.objective_names:
            study.set_user_attr("objective_names", list(args.objective_names))
    except Exception as exc:
        logger.warning("Failed to persist objective names on study '%s': %s", args.study_name, exc)

    # --- 2. Main Optimization Loop ---
    num_optuna_super_batches = (
        args.n_trials + args.optuna_study_batch_size - 1
    ) // args.optuna_study_batch_size

    for super_batch_idx in range(num_optuna_super_batches):
        logger.info(
            f"\n--- Starting Optuna Super-Batch {super_batch_idx + 1}/{num_optuna_super_batches} ---"
        )

        # Ask Optuna for a batch of trials
        optuna_trials_in_super_batch: list[optuna.Trial] = []
        suggested_params_for_super_batch: list[dict[str, Any] | None] = []

        num_trials_this_super_batch = min(
            args.optuna_study_batch_size,
            args.n_trials - (super_batch_idx * args.optuna_study_batch_size),
        )
        if num_trials_this_super_batch <= 0:
            break

        for _ in range(num_trials_this_super_batch):
            trial_obj = study.ask()
            optuna_trials_in_super_batch.append(trial_obj)
            current_trial_params = {}
            try:
                for name, dist_obj in hyperparam_distributions.items():
                    current_trial_params[name] = _suggest_from_dist(trial_obj, name, dist_obj)
                suggested_params_for_super_batch.append(current_trial_params)
            except optuna.exceptions.TrialPruned:
                logger.info(
                    f"Trial {trial_obj.number} pruned during suggestion phase. Skipping execution."
                )
                suggested_params_for_super_batch.append(None)  # Mark as pruned

        # Filter out trials that were pruned
        valid_trials = [
            t
            for t, p in zip(
                optuna_trials_in_super_batch, suggested_params_for_super_batch, strict=False
            )
            if p is not None
        ]
        valid_params = [p for p in suggested_params_for_super_batch if p is not None]
        if not valid_trials:
            logger.info("No valid trials to execute in this super-batch.")
            continue

        # Format for the vectorized runner
        predefined_samples_for_runner = defaultdict(list)
        for trial_obj, params_dict in zip(valid_trials, valid_params, strict=False):
            for k, v in params_dict.items():
                predefined_samples_for_runner[k].append(v)
            predefined_samples_for_runner[sample_id_key].append(trial_obj.number)

        predefined_samples_for_runner = dict(predefined_samples_for_runner)
        num_valid_samples = len(predefined_samples_for_runner[sample_id_key])

        # Create a unique output directory for this execution batch within the main study output
        current_batch_output_dir = output_dir / f"optuna_exec_batch_{super_batch_idx:04d}"
        current_batch_output_dir.mkdir(parents=True, exist_ok=True)

        # Create a SamplingSweepConfig on-the-fly for the runner function
        runner_args = SamplingSweepConfig(
            algo_and_network_config=args.algo_and_network_config,
            env_config=args.env_config,
            output_root=current_batch_output_dir,  # Use the batch-specific dir
            seed=(args.seed + super_batch_idx if args.seed is not None else super_batch_idx),
            hparam_batch_size=args.runner_hparam_batch_size,
            num_samples=num_valid_samples,
            experiment_type="optuna_vectorized_batch",
            sampling_method="predefined",
            min_hparam_batch_size=1,
            resume=False,
            run_length_modifier=args.run_length_modifier,
            log_level=args.log_level,
            save_logs=args.save_logs,
        )
        runner_args.output_dir = current_batch_output_dir  # Manually set the final dir

        # Create a config instance for this specific execution batch
        batch_exp_config_instance = copy.deepcopy(base_config_template)

        new_logger_config = dataclasses.replace(
            batch_exp_config_instance.logger,
            base_exp_path=str(current_batch_output_dir),
        )
        batch_exp_config_instance.logger = new_logger_config
        batch_exp_config_instance.optuna_objective_names_for_runner = args.objective_names

        exp_config_container_for_runner = AlgoSpecificExperimentConfigContainer(
            algo_name=exp_info.algo_name,
            experiment_config=batch_exp_config_instance,
            hyperparam_dist_config=hyperparam_distributions,
            algo_main_fn=algorithm_main_fn,
            hyperparams_container_spec=None,
        )

        # Execute the batch of trials
        run_metrics_by_sample_id: dict[int, dict[str, float]] = {}
        try:
            _, run_metrics_by_sample_id = process_batched_sample_set(
                args=runner_args,
                exp_config_container=exp_config_container_for_runner,
                predefined_samples=predefined_samples_for_runner,
            )
        except Exception as e:
            logger.error(f"Execution of super-batch {super_batch_idx + 1} failed: {e}")
            traceback.print_exc()
            # Mark all running trials in this batch as FAIL
            for trial_obj in valid_trials:
                study.tell(trial_obj, state=optuna.trial.TrialState.FAIL)

        # Report results back to Optuna
        for trial_obj in valid_trials:
            trial_num = trial_obj.number
            if trial_num in run_metrics_by_sample_id:
                trial_metrics = run_metrics_by_sample_id[trial_num]
                values_to_report = [
                    float(trial_metrics.get(name, 0.0)) for name in args.objective_names
                ]
                study.tell(trial_obj, values_to_report)
                log_msg = ", ".join(
                    [
                        f"{name}={val:.4f}"
                        for name, val in zip(args.objective_names, values_to_report, strict=False)
                    ]
                )
                logger.info(f"Trial {trial_num} finished. Objectives -> {log_msg}")
            else:
                logger.warning(f"Metrics for trial {trial_num} not found. Reporting as FAIL.")
                study.tell(trial_obj, state=optuna.trial.TrialState.FAIL)

    # --- 5. Final Optuna Study Summary & Analysis Plot Saving ---
    logger.info("\n--- Optuna Study Complete ---")
    logger.info(f"Number of finished trials in study: {len(study.trials)}")

    # --- Save Best Parameters ---
    if len(args.objective_directions) > 1:
        logger.info("Pareto front (non-dominated trials):")
        try:
            best_trials_list = study.best_trials
            if not best_trials_list:
                logger.warning(
                    "No completed trials found, or Pareto front could not be determined (study.best_trials is empty)."
                )
            else:
                for i, best_trial_obj in enumerate(best_trials_list):
                    logger.info(
                        f"  Trial {i + 1} on Pareto Front (Number {best_trial_obj.number}):"
                    )
                    objective_values_str = ", ".join(
                        [
                            f"{name}={val:.4f}"
                            for name, val in zip(
                                args.objective_names, best_trial_obj.values, strict=False
                            )
                        ]
                    )
                    logger.info(f"    Values: {objective_values_str}")
                    logger.info(f"    Params: {best_trial_obj.params}")
                pareto_params_list = [
                    {"trial_number": t.number, "values": t.values, "params": t.params}
                    for t in best_trials_list
                ]
                pareto_params_path = output_dir / "pareto_front_params.json"
                with open(pareto_params_path, "w") as f:
                    json.dump(pareto_params_list, f, indent=4)
                logger.info(
                    f"Parameters and values for all trials on the Pareto front saved to {pareto_params_path}"
                )
        except ValueError as ve:
            logger.warning(
                f"Could not retrieve Pareto front (Optuna.study.best_trials error): {ve}"
            )
        except Exception as e:
            logger.error(f"Error retrieving Pareto front information: {e}")
            traceback.print_exc()
    else:  # Single objective case
        try:
            best_overall_trial = study.best_trial
            logger.info("Best trial overall:")
            logger.info(f"  Value ({args.objective_names[0]}): {best_overall_trial.value:.4f}")
            logger.info(f"  Params: {best_overall_trial.params}")
            logger.info(f"  Trial Number: {best_overall_trial.number}")
            best_params_path = output_dir / "best_optuna_params.json"
            with open(best_params_path, "w") as f:
                json.dump(best_overall_trial.params, f, indent=4)
            logger.info(f"Best Optuna parameters saved to {best_params_path}")
        except ValueError:
            logger.warning("No completed trials found, or best_trial could not be determined.")
        except Exception as e:
            logger.error(f"Error retrieving best trial information: {e}")
            traceback.print_exc()

    # --- Save Analysis Plots if Requested ---
    if args.save_optuna_plots:
        optuna_plots_dir.mkdir(parents=True, exist_ok=True)
        logger.info(f"Saving Optuna analysis plots to: {optuna_plots_dir}")
        completed_trials = [t for t in study.trials if t.state == optuna.trial.TrialState.COMPLETE]

        if not completed_trials:
            logger.warning("No completed trials found. Skipping analysis plot generation.")
        else:
            try:
                # Optimization History (one plot per objective if multi-objective)
                if len(args.objective_names) == 1:
                    fig_history = plot_optimization_history(study)
                    fig_history.write_html(str(optuna_plots_dir / "optimization_history.html"))
                else:
                    for i, obj_name in enumerate(args.objective_names):
                        fig_history = plot_optimization_history(
                            study, target_name=obj_name, target=lambda t: t.values[i]
                        )
                        fig_history.write_html(
                            str(optuna_plots_dir / f"optimization_history_{obj_name}.html")
                        )
                logger.info("Saved: Optimization History Plot")

                # Parameter Importances (one plot per objective if multi-objective)
                if len(args.objective_names) == 1:
                    fig_importance = plot_param_importances(study)
                    fig_importance.write_html(str(optuna_plots_dir / "param_importances.html"))
                else:
                    for i, obj_name in enumerate(args.objective_names):
                        fig_importance = plot_param_importances(
                            study, target_name=obj_name, target=lambda t: t.values[i]
                        )
                        fig_importance.write_html(
                            str(optuna_plots_dir / f"param_importances_{obj_name}.html")
                        )
                logger.info("Saved: Parameter Importances Plot")

                # Parallel Coordinate Plot (one plot per objective if multi-objective)
                if len(args.objective_names) == 1:
                    fig_parallel = plot_parallel_coordinate(study)
                    fig_parallel.write_html(str(optuna_plots_dir / "parallel_coordinate.html"))
                else:
                    for i, obj_name in enumerate(args.objective_names):
                        fig_parallel = plot_parallel_coordinate(
                            study, target_name=obj_name, target=lambda t: t.values[i]
                        )
                        fig_parallel.write_html(
                            str(optuna_plots_dir / f"parallel_coordinate_{obj_name}.html")
                        )
                logger.info("Saved: Parallel Coordinate Plot")

                # Slice Plot (requires specifying params, one plot per objective if multi-objective)
                # You might want to select only a few important params for slice plots
                param_names_for_slice = (
                    [param_name for param_name in study.best_trials[0].params.keys()]
                    if study.best_trials
                    else []
                )  # Get from a trial
                if param_names_for_slice:
                    if len(args.objective_names) == 1:
                        fig_slice = plot_slice(study, params=param_names_for_slice)
                        fig_slice.write_html(str(optuna_plots_dir / "slice_plot.html"))
                    else:
                        for i, obj_name in enumerate(args.objective_names):
                            fig_slice = plot_slice(
                                study,
                                params=param_names_for_slice,
                                target_name=obj_name,
                                target=lambda t: t.values[i],
                            )
                            fig_slice.write_html(
                                str(optuna_plots_dir / f"slice_plot_{obj_name}.html")
                            )
                    logger.info("Saved: Slice Plot")

                # Contour Plot (requires specifying pairs of params)
                # Example: plot contour for the first two params if available
                if len(param_names_for_slice) >= 2:
                    if len(args.objective_names) == 1:
                        fig_contour = plot_contour(study, params=param_names_for_slice[:2])
                        fig_contour.write_html(str(optuna_plots_dir / "contour_plot.html"))
                    else:
                        for i, obj_name in enumerate(args.objective_names):
                            fig_contour = plot_contour(
                                study,
                                params=param_names_for_slice[:2],
                                target_name=obj_name,
                                target=lambda t: t.values[i],
                            )
                            fig_contour.write_html(
                                str(optuna_plots_dir / f"contour_plot_{obj_name}.html")
                            )
                    logger.info("Saved: Contour Plot (for first two params)")

                # Pareto Front Plot (only for multi-objective)
                if len(args.objective_directions) > 1:
                    fig_pareto = plot_pareto_front(study, target_names=args.objective_names)
                    fig_pareto.write_html(str(optuna_plots_dir / "pareto_front.html"))
                    logger.info("Saved: Pareto Front Plot")

            except Exception as e_plot:
                logger.error(f"Failed to generate or save one or more analysis plots: {e_plot}")
                traceback.print_exc()
