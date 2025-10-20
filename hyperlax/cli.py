import logging
import sys
import time
import traceback
import typing
from datetime import timedelta

import jax
import tyro

from hyperlax.runner.launch import (
    BenchmarkRunConfig,
    GenerateSamplesConfig,
    OptunaSweepConfig,
    PlotBenchmarkConfig,
    SamplingSweepConfig,
    SingleRunConfig,
    launch_benchmark,
    launch_optuna_sweep,
    launch_plot_benchmark,
    launch_sample_generation,
    launch_sampling_sweep,
    launch_single_run,
)
from hyperlax.runner.launcher_utils import setup_logging
from hyperlax.utils.jax_utils import print_xla_env_vars

logger = logging.getLogger(__name__)


Commands = tyro.conf.FlagConversionOff[
    typing.Union[
        typing.Annotated[
            SamplingSweepConfig,
            tyro.conf.subcommand(
                name="sweep-hp-samples",
                description="Run a hyperparameter sweep via our sampling strategies (qmc_sobol, random, etc.).",
            ),
        ],
        typing.Annotated[
            OptunaSweepConfig,
            tyro.conf.subcommand(
                name="optuna-hp-search",
                description="Run a hyperparameter optimization study with Optuna.",
            ),
        ],
        typing.Annotated[
            SingleRunConfig,
            tyro.conf.subcommand(
                name="run-single-hp", description="Run a single hyperparam experiment."
            ),
        ],
        typing.Annotated[
            GenerateSamplesConfig,
            tyro.conf.subcommand(
                name="generate-hp-samples",
                description="Generate and save a hyperparameter workload file.",
            ),
        ],
        typing.Annotated[
            BenchmarkRunConfig,
            tyro.conf.subcommand(
                name="run-benchmark",
                description="Run a benchmark suite from a configuration file.",
            ),
        ],
        typing.Annotated[
            PlotBenchmarkConfig,
            tyro.conf.subcommand(
                name="plot-benchmark",
                description="Analyze and plot results from a benchmark run.",
            ),
        ],
    ]
]


def main() -> None:
    """Main entry point for all hyperlax experiments."""
    try:
        cmd = tyro.cli(Commands)
        setup_logging(cmd.log_level)
        logger.info("--- hyperlax Experiment Launcher ---")
        print_xla_env_vars()
        start_time = time.time()

        print("JAX backend:", jax.default_backend())
        print("Devices:", jax.devices())

        if isinstance(cmd, SamplingSweepConfig):
            print(">>> Running Sampling Sweep <<<")
            launch_sampling_sweep(cmd)
        elif isinstance(cmd, OptunaSweepConfig):
            print(">>> Running Optuna Sweep <<<")
            launch_optuna_sweep(cmd)
        elif isinstance(cmd, SingleRunConfig):
            print(">>> Running Single Experiment <<<")
            launch_single_run(cmd)
        elif isinstance(cmd, GenerateSamplesConfig):
            print(">>> Generating Hyperparameter Samples <<<")
            launch_sample_generation(cmd)
        elif isinstance(cmd, BenchmarkRunConfig):
            print(">>> Running Benchmark Suite <<<")
            launch_benchmark(cmd)
        elif isinstance(cmd, PlotBenchmarkConfig):
            print(">>> Plotting Benchmark Results <<<")
            launch_plot_benchmark(cmd)
        else:
            raise ValueError(f"Unknown command type: {type(cmd)}")

        elapsed = round(time.time() - start_time)
        td = timedelta(seconds=elapsed)
        days = td.days
        hours, remainder = divmod(td.seconds, 3600)
        minutes, seconds = divmod(remainder, 60)
        formatted_duration = f"{days}-{hours:02}:{minutes:02}:{seconds:02}"
        print(f"=== Total execution time of hyperlax/cli.py (D-HH:MM:SS): {formatted_duration} ===")

    except Exception:
        print("\n--- FATAL ERROR in hyperlax.cli ---", file=sys.stderr)
        traceback.print_exc(file=sys.stderr)
        sys.exit(1)


if __name__ == "__main__":
    main()
