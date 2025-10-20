import dataclasses
import logging
import sys
import time
import traceback
from dataclasses import fields, replace
from datetime import timedelta
from pathlib import Path

from ..analysis._main import run_analysis
from hyperlax.runner.fs_utils import (
    is_experiment_complete,
    mark_experiment_complete,
    save_args_config_and_metadata,
)
from hyperlax.runner.launch_args import (
    BenchmarkRunConfig,
    GenerateSamplesConfig,
    OptunaSweepConfig,
    PlotBenchmarkConfig,
    SamplingSweepConfig,
    SingleRunConfig,
)
from hyperlax.runner.launcher_utils import (
    build_main_experiment_config,
    create_single_hp_batch_from_config_defaults,
    load_benchmark_config,
    setup_output_directory,
)
from hyperlax.runner.optuna import run_optuna_sweep_impl
from hyperlax.runner.sample_generation import generate_hyperparam_sample_and_save
from hyperlax.runner.sampling import (
    process_batched_sample_set,
    process_sequential_sample_set,
)

logger = logging.getLogger(__name__)


def launch_sampling_sweep(args: "SamplingSweepConfig") -> None:
    """Orchestrates a sampling-based hyperparameter sweep."""
    if args.resume_from:
        resume_path = Path(args.resume_from)
        if not resume_path.is_dir():
            logger.error(f"Resume path specified but does not exist or is not a directory: {resume_path}")
            raise FileNotFoundError(f"Resume path not found: {resume_path}")
        args.output_root = resume_path
        args.resume = True
        logger.info(f"Resuming from: {args.resume_from}. Setting output_root and enabling resume.")

    args.output_dir = setup_output_directory(args)
    # Metadata is saved once for the parent sweep.
    if not args.output_dir.joinpath("args.json").exists():
        save_args_config_and_metadata(args, args.output_dir)
    logger.info(f"Output directory for sweep: {args.output_dir.resolve()}")

    exp_info = build_main_experiment_config(args)

    # Adjust logger path
    new_logger_config = dataclasses.replace(
        exp_info.experiment_config.logger,
        base_exp_path=str(args.output_dir),
        save_console_to_file=args.save_logs,
    )
    exp_info = dataclasses.replace(
        exp_info,
        experiment_config=dataclasses.replace(exp_info.experiment_config, logger=new_logger_config),
    )

    if args.sequential:
        if args.group_by_structural_hparams:
            logger.warning("=--group-by-structural-hparams= is ignored when =--sequential= is used.")
        process_sequential_sample_set(args, exp_info)
        return

    if args.group_by_structural_hparams:
        logger.info("Running in HOMOGENEOUS BATCHED mode (grouping by structural params).")
    else:
        logger.info("Running in standard HETEROGENEOUS BATCHED mode.")
    process_batched_sample_set(args=args, exp_config_container=exp_info)


def launch_optuna_sweep(args: "OptunaSweepConfig") -> None:
    """Sets up the config and output dir, then calls the dedicated Optuna runner."""
    if args.resume_from:
        resume_path = Path(args.resume_from)
        if not resume_path.is_dir():
            logger.error(f"Resume path specified but does not exist or is not a directory: {resume_path}")
            raise FileNotFoundError(f"Resume path not found: {resume_path}")
        args.output_root = resume_path
        args.resume = True
        logger.info(f"Resuming from: {args.resume_from}. Setting output_root and enabling resume.")

    args.output_dir = setup_output_directory(args)
    save_args_config_and_metadata(args, args.output_dir)
    logger.info(f"Output directory for Optuna study: {args.output_dir.resolve()}")

    # Build the experiment container which holds the base config, dists, spec, etc.
    exp_info = build_main_experiment_config(args)
    new_logger_config = replace(
        exp_info.experiment_config.logger,
        base_exp_path=str(args.output_dir),
        save_console_to_file=args.save_logs,
    )
    exp_info.experiment_config.logger = new_logger_config
    run_optuna_sweep_impl(args, exp_info)


def launch_single_run(args: "SingleRunConfig") -> None:
    """Orchestrates a single, non-sweep experiment run."""
    args.output_dir = setup_output_directory(args)
    save_args_config_and_metadata(args, args.output_dir)
    logger.info(f"Output directory for single run: {args.output_dir.resolve()}")

    exp_info = build_main_experiment_config(args)
    original_config = exp_info.experiment_config

    # For a single run, we create a "dummy" batch of size 1 using the default values from the config.
    # This makes the single run compatible with the batched runner (`run_experiment`).
    single_hp_batch = create_single_hp_batch_from_config_defaults(original_config)

    new_logger_config = replace(original_config.logger, base_exp_path=str(args.output_dir))
    new_training_config = replace(
        original_config.training,
        hyperparam_batch_enabled=True,  # The runner expects this to be True
        hyperparam_batch_size=1,
        hyperparam_batch_samples=single_hp_batch,
        hyperparam_batch_sample_ids=[0],  # sample_id 0 for this single run
    )
    config_single = replace(
        original_config,
        experiment_mode="single",
        training=new_training_config,
        logger=new_logger_config,
    )
    exp_info = replace(exp_info, experiment_config=config_single)

    logger.info("Launching single experiment runner...")
    # The algorithm's main function is called directly with the configured object.
    exp_info.algo_main_fn(config_single)


def launch_sample_generation(args: "GenerateSamplesConfig") -> None:
    """Generates and saves a hyperparameter workload file."""
    exp_info = build_main_experiment_config(args, skip_env_setup=True)
    new_logger_config = replace(exp_info.experiment_config.logger, save_console_to_file=args.save_logs)
    exp_info.experiment_config.logger = new_logger_config

    # Delegate to the generation logic
    generate_hyperparam_sample_and_save(
        output_file=Path(args.output_file),
        num_samples=args.num_samples,
        sampling_method=args.sampling_method,
        seed=args.seed,
        exp_config_container=exp_info,
    )

def launch_plot_benchmark(args: "PlotBenchmarkConfig") -> None:
    """Analyzes and plots results from a completed benchmark run."""
    main_output_dir = Path(args.results_dir_to_plot)
    if not main_output_dir.exists():
        logger.error(f"Specified results_dir_to_plot does not exist: {main_output_dir}")
        return

    logger.info(f"Plotting results from benchmark directory: {main_output_dir}")

    # Exclude common non-result directories
    all_result_dirs = [
        d for d in main_output_dir.iterdir() if d.is_dir() and d.name not in ["slurm_logs", "_plots", "optuna_plots"]
    ]

    # Create a dedicated directory for analysis plots
    plots_output_dir = main_output_dir.parent / f"{main_output_dir.name}_plots"
    plots_output_dir.mkdir(parents=True, exist_ok=True)
    logger.info(f"Analysis plots will be saved to: {plots_output_dir.resolve()}")

    plot_args_dict = dataclasses.asdict(args)

    if all_result_dirs:
        run_analysis(
            result_dirs=all_result_dirs,
            output_dir=plots_output_dir,
            include_algo_variants=args.include_additional_info_in_titles,
            top_n=args.top_n,
            slice_metric=args.slice_metric,
            max_workers=args.max_workers,
            benchmark_root=main_output_dir,
            plot_config=plot_args_dict,
            combined_only=args.combined_only,
        )
    else:
        logger.error(f"No result directories found under {main_output_dir} to plot.")


def launch_benchmark(args: "BenchmarkRunConfig") -> None:
    """Run or plot benchmark directly using BenchmarkRunConfig, supporting loading from config file."""
    logger.debug(f"Benchmark workflow started with args: {args}")

    if args.resume_from:
        resume_path = Path(args.resume_from)
        if not resume_path.is_dir():
            logger.error(f"Resume path specified but does not exist or is not a directory: {resume_path}")
            raise FileNotFoundError(f"Resume path not found: {resume_path}")
        args.output_root = resume_path
        args.resume = True
        logger.info(f"Resuming from: {args.resume_from}. Setting output_root and enabling resume.")

    # Load benchmark config from file if specified, else use CLI args as base
    if args.base_config:
        try:
            base_cfg = load_benchmark_config(args.base_config)
        except Exception:
            logger.error(f"Failed to load benchmark config: {args.base_config}")
            sys.exit(1)
    else:
        # Use CLI args directly as base config
        base_cfg = args

    default_args = {}
    for f in fields(BenchmarkRunConfig):
        default_args[f.name] = f.default

    base_cfg_dict = dataclasses.asdict(base_cfg)
    cli_args_dict = dataclasses.asdict(args)

    # Override only if CLI arg differs from default value to avoid overwriting loaded config unintentionally
    for key, cli_val in cli_args_dict.items():
        default_val = default_args.get(key, None)
        if cli_val is not None and cli_val != default_val:
            base_cfg_dict[key] = cli_val

    benchmark_cfg = BenchmarkRunConfig(**base_cfg_dict)

    if benchmark_cfg.resume:
        main_output_dir = Path(benchmark_cfg.output_root)
        logger.info(f"Resuming benchmark in directory: {main_output_dir.resolve()}")
    else:
        timestamp = time.strftime("%Y%m%d-%H%M%S")
        main_output_dir = Path(benchmark_cfg.output_root) / f"{benchmark_cfg.benchmark_name or 'benchmark'}_{timestamp}"
    main_output_dir.mkdir(parents=True, exist_ok=True)
    logger.debug(f"Main benchmark output directory: {main_output_dir.resolve()}")

    if not benchmark_cfg.resume:
        print(f"Benchmark arguments to execute: {benchmark_cfg}")
        save_args_config_and_metadata(benchmark_cfg, main_output_dir, save_requirement_txt=True)

    all_result_dirs = []
    for algo in benchmark_cfg.algos or []:
        for env in benchmark_cfg.envs or []:
            for mode in benchmark_cfg.sweep_modes or []:
                # Construct run_dir_name first for logging and checks
                algo_part = algo.replace("_", "-")
                env_part = env
                suffix_part = f"{mode}-N{benchmark_cfg.num_samples_per_run}-{benchmark_cfg.run_length_modifier}"
                run_dir_name = f"{algo_part}_{env_part}_{suffix_part}"
                run_output_root = main_output_dir / run_dir_name

                if benchmark_cfg.resume and is_experiment_complete(run_output_root):
                    logger.info(f"--- Skipping Completed Benchmark: {run_dir_name} ---")
                    all_result_dirs.append(run_output_root)
                    continue

                try:
                    # --- PRE-FLIGHT CHECK for algo+env compatibility ---
                    # This will raise NotImplementedError or ImportError if invalid.
                    @dataclasses.dataclass
                    class TmpArgs:
                        algo_and_network_config: str
                        env_config: str
                        run_length_modifier: str

                    tmp_args = TmpArgs(
                        algo_and_network_config=algo,
                        env_config=env,
                        run_length_modifier=benchmark_cfg.run_length_modifier,
                    )
                    build_main_experiment_config(tmp_args)

                    # --- EXECUTION ---
                    print(f"--- Starting Benchmark: {algo}/{env}/{mode} ---")
                    start_time = time.time()
                    run_output_root.mkdir(parents=True, exist_ok=True)

                    base_args_for_run = {
                        "algo_and_network_config": algo,
                        "env_config": env,
                        "run_length_modifier": benchmark_cfg.run_length_modifier,
                        "output_root": run_output_root,
                        "seed": benchmark_cfg.seed,
                        "log_level": benchmark_cfg.log_level,
                        "save_logs": True,
                    }

                    if mode in ["sequential", "batched"]:
                        sampled_hyperparams_file = run_output_root / "sampled_hyperparams.csv"
                        if not sampled_hyperparams_file.exists():
                            # Only pass relevant args to GenerateSamplesConfig
                            gen_args_dict = base_args_for_run.copy()
                            gen_args_dict.update(
                                {
                                    "output_file": str(sampled_hyperparams_file),
                                    "num_samples": benchmark_cfg.num_samples_per_run,
                                }
                            )
                            gen_args = GenerateSamplesConfig(**gen_args_dict)
                            launch_sample_generation(gen_args)

                        # Pass the resume flag to SamplingSweepConfig
                        sample_args_dict = base_args_for_run.copy()
                        sample_args_dict.update(
                            {
                                "num_samples": benchmark_cfg.num_samples_per_run,
                                "load_hyperparams": str(sampled_hyperparams_file),
                                "sequential": (mode == "sequential"),
                                "group_by_structural_hparams": (
                                    (mode == "batched") and benchmark_cfg.group_by_structural_hparams
                                ),
                                "hparam_batch_size": benchmark_cfg.hparam_batch_size,
                                "min_hparam_batch_size": 1,
                                "resume": benchmark_cfg.resume,  # Pass resume flag here
                            }
                        )
                        sample_args = SamplingSweepConfig(**sample_args_dict)
                        sample_args.output_dir = run_output_root  # Manually set
                        launch_sampling_sweep(sample_args)

                    elif mode == "optuna":
                        # Pass the resume flag to OptunaSweepConfig
                        optuna_args_dict = base_args_for_run.copy()
                        optuna_args_dict.update(
                            {
                                "n_trials": benchmark_cfg.num_samples_per_run,
                                "runner_hparam_batch_size": benchmark_cfg.hparam_batch_size,
                                "resume": benchmark_cfg.resume,  # Pass resume flag here
                            }
                        )
                        optuna_args = OptunaSweepConfig(**optuna_args_dict)
                        optuna_args.output_dir = run_output_root
                        launch_optuna_sweep(optuna_args)
                    else:
                        raise ValueError(f"Unsupported benchmark mode: {mode}")

                    mark_experiment_complete(run_output_root)
                    all_result_dirs.append(run_output_root)

                    elapsed = round(time.time() - start_time)
                    td = timedelta(seconds=elapsed)
                    days = td.days
                    hours, remainder = divmod(td.seconds, 3600)
                    minutes, seconds = divmod(remainder, 60)
                    formatted_duration = f"{days}-{hours:02}:{minutes:02}:{seconds:02}"
                    print(f"--- Finished Benchmark: {run_dir_name} (in {formatted_duration}) ---")

                except NotImplementedError as e:
                    logger.warning(f"--- Skipping Unsupported Combination: {run_dir_name} ---")
                    logger.warning(f"Reason: {e}")
                    continue
                except ImportError as e:
                    logger.warning(f"--- Skipping Invalid/Missing Config: {run_dir_name} ---")
                    logger.warning(f"Reason: {e}")
                    continue
                except Exception:
                    print(
                        f"\n--- ERROR during Benchmark: {algo}/{env}/{mode} ---",
                        file=sys.stderr,
                    )
                    traceback.print_exc(file=sys.stderr)
                    # Continue to the next benchmark run instead of exiting
                    logger.error(f"Benchmark run {algo}/{env}/{mode} failed. Continuing to next run.")

    if all_result_dirs:
        plots_output_dir = main_output_dir.parent / f"{main_output_dir.name}_plots"
        plots_output_dir.mkdir(parents=True, exist_ok=True)
        default_plot_config = {
            "results_dir_to_plot": main_output_dir,
            "include_additional_info_in_titles": True,
            "top_n": 5,
            "slice_metric": "peak_return",
            "max_workers": None,
        }
        run_analysis(
            result_dirs=all_result_dirs,
            output_dir=plots_output_dir,
            include_algo_variants=True,
            top_n=5,
            benchmark_root=main_output_dir,
            plot_config=default_plot_config,
        )
    else:
        logger.error("No benchmark runs completed successfully.")
