import sys
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any
from unittest.mock import MagicMock

import pytest
from hyperlax.cli import main
from hyperlax.configs.env.base_env import BaseEnvironmentConfig
from hyperlax.configs.logger.base_logger import LoggerConfig
from hyperlax.configs.main_base import BaseExperimentConfig
from hyperlax.configs.training.base_training import BaseTrainingConfig
from hyperlax.hyperparam.tunable import Tunable
from hyperlax.runner.base_types import AlgoSpecificExperimentConfigContainer
from hyperlax.runner.launch import (
    launch_optuna_sweep,
    launch_plot_benchmark,
    launch_sample_generation,
    launch_sampling_sweep,
    launch_single_run,
)
from hyperlax.runner.launch_args import (
    BenchmarkRunConfig,
    GenerateSamplesConfig,
    OptunaSweepConfig,
    PlotBenchmarkConfig,
    SamplingSweepConfig,
    SingleRunConfig,
)


@dataclass
class MockNetworkConfig:
    # This can be empty, it just needs to exist.
    pass


@dataclass
class MockAlgoConfig:
    _target_: str = "mock.algo.main"
    hyperparam: Any = None
    network: Any = field(default_factory=MockNetworkConfig)


@dataclass
class MockHyperparams:
    sample_id: Tunable = field(
        default_factory=lambda: Tunable(value=-1, is_vectorized=True, is_fixed=True, expected_type=int)
    )


@pytest.fixture
def mock_exp_container(monkeypatch):
    """Provides a mock AlgoSpecificExperimentConfigContainer with real dataclasses."""
    mock_logger_config = LoggerConfig(base_exp_path="mock/path")
    mock_training_config = BaseTrainingConfig(hyperparam_batch_enabled=False)
    mock_algo_config = MockAlgoConfig(hyperparam=MockHyperparams())
    mock_env_config = BaseEnvironmentConfig(env_name="mock_env")

    mock_config = BaseExperimentConfig(
        algorithm=mock_algo_config,
        env=mock_env_config,
        training=mock_training_config,
        logger=mock_logger_config,
        config_name="mock_config",
    )

    mock_main_fn = MagicMock()

    container = AlgoSpecificExperimentConfigContainer(
        algo_name="mock_algo",
        experiment_config=mock_config,
        hyperparam_dist_config={},
        algo_main_fn=mock_main_fn,
        hyperparams_container_spec=None,
    )

    monkeypatch.setattr(
        "hyperlax.runner.launch.build_main_experiment_config",
        lambda *args, **kwargs: container,
    )
    return container


@pytest.fixture
def mock_launch_dependencies(monkeypatch, mock_exp_container):
    """Mocks all heavy-lifting functions called directly by the launch scripts."""
    mocks_to_patch = {
        "hyperlax.runner.launch.process_sequential_sample_set": MagicMock(),
        "hyperlax.runner.launch.process_batched_sample_set": MagicMock(),
        "hyperlax.runner.launch.run_optuna_sweep_impl": MagicMock(),
        "hyperlax.runner.launch.generate_hyperparam_sample_and_save": MagicMock(),
        "hyperlax.runner.launch.load_benchmark_config": MagicMock(
            return_value=BenchmarkRunConfig(
                algos=["mock_algo"],
                envs=["mock_env"],
                sweep_modes=["sequential"],
                num_samples_per_run=1,
            )
        ),
    }

    mock_objects = {}
    for path, mock_obj in mocks_to_patch.items():
        monkeypatch.setattr(path, mock_obj)
        simple_name = path.split(".")[-1]
        mock_objects[simple_name] = mock_obj

    mock_objects["algo_main_fn"] = mock_exp_container.algo_main_fn
    return mock_objects


# --- Test Single Run ---
def test_launch_single_run_creates_dir_and_calls_main(tmp_path, mock_launch_dependencies):
    mocks = mock_launch_dependencies
    args = SingleRunConfig(
        algo_and_network_config="mock_algo",
        env_config="mock_env",
        output_root=tmp_path,
    )

    launch_single_run(args)

    run_dir = tmp_path / "mock_algo" / "mock_env"
    assert run_dir.exists()
    assert (run_dir / "args.json").exists()

    mocks["algo_main_fn"].assert_called_once()
    called_config = mocks["algo_main_fn"].call_args[0][0]

    assert called_config.training.hyperparam_batch_enabled is True
    assert called_config.training.hyperparam_batch_size == 1
    assert called_config.experiment_mode == "single"


# --- Test Sampling Sweep ---
def test_launch_sampling_sweep_sequential(tmp_path, mock_launch_dependencies):
    mocks = mock_launch_dependencies
    args = SamplingSweepConfig(output_root=tmp_path, sequential=True)
    launch_sampling_sweep(args)
    mocks["process_sequential_sample_set"].assert_called_once()
    mocks["process_batched_sample_set"].assert_not_called()


def test_launch_sampling_sweep_batched(tmp_path, mock_launch_dependencies):
    mocks = mock_launch_dependencies
    args = SamplingSweepConfig(output_root=tmp_path, sequential=False)
    launch_sampling_sweep(args)
    mocks["process_batched_sample_set"].assert_called_once()
    mocks["process_sequential_sample_set"].assert_not_called()


@pytest.mark.parametrize(
    "config_class, launch_fn",
    [
        (SamplingSweepConfig, launch_sampling_sweep),
        (OptunaSweepConfig, launch_optuna_sweep),
    ],
)
def test_sweep_resume_from_flag(tmp_path, config_class, launch_fn, mock_launch_dependencies):
    resume_dir = tmp_path / "existing_run"
    resume_dir.mkdir()
    args = config_class(resume_from=str(resume_dir))
    launch_fn(args)
    assert args.output_root == resume_dir
    assert args.resume is True
    assert args.output_dir == resume_dir


def test_sweep_resume_from_nonexistent_path(tmp_path):
    non_existent_path = tmp_path / "non_existent"
    args = SamplingSweepConfig(resume_from=str(non_existent_path))
    with pytest.raises(FileNotFoundError):
        launch_sampling_sweep(args)


# --- Test Optuna Sweep ---
def test_launch_optuna_sweep(tmp_path, mock_launch_dependencies, mock_exp_container):
    mocks = mock_launch_dependencies
    args = OptunaSweepConfig(output_root=tmp_path)
    launch_optuna_sweep(args)
    mocks["run_optuna_sweep_impl"].assert_called_once_with(args, mock_exp_container)


# --- Test Sample Generation ---
def test_launch_sample_generation(tmp_path, mock_launch_dependencies, mock_exp_container):
    mocks = mock_launch_dependencies
    output_file = tmp_path / "samples.csv"
    args = GenerateSamplesConfig(output_file=str(output_file), num_samples=128)
    launch_sample_generation(args)
    mocks["generate_hyperparam_sample_and_save"].assert_called_once()
    call_kwargs = mocks["generate_hyperparam_sample_and_save"].call_args.kwargs
    assert call_kwargs["output_file"] == output_file
    assert call_kwargs["num_samples"] == 128
    assert call_kwargs["exp_config_container"] == mock_exp_container


# --- Test Analysis and Plotting ---


@pytest.fixture
def mock_launchers(mocker):
    """Mocks all launch_* functions in hyperlax.cli to test CLI dispatching."""
    return {
        "launch_sampling_sweep": mocker.patch("hyperlax.cli.launch_sampling_sweep"),
        "launch_optuna_sweep": mocker.patch("hyperlax.cli.launch_optuna_sweep"),
        "launch_single_run": mocker.patch("hyperlax.cli.launch_single_run"),
        "launch_sample_generation": mocker.patch("hyperlax.cli.launch_sample_generation"),
        "launch_benchmark": mocker.patch("hyperlax.cli.launch_benchmark"),
        "launch_plot_benchmark": mocker.patch("hyperlax.cli.launch_plot_benchmark"),
    }


def run_cli_main(command_str: str, mocker):
    """Parses a command string and runs the main CLI entrypoint."""
    sys.argv = ["hyperlax/cli.py"] + command_str.split()
    mocker.patch("sys.exit")  # Prevent test runner from exiting
    main()


@pytest.mark.parametrize(
    "command, expected_launcher, expected_config_type, expected_attrs",
    [
        # --- Test Single Run ---
        (
            "run-single-hp --algo-and-network-config ppo_mlp --env-config gymnax.pendulum --output-root /tmp/single",
            "launch_single_run",
            SingleRunConfig,
            {
                "algo_and_network_config": "ppo_mlp",
                "env_config": "gymnax.pendulum",
                "output_root": Path("/tmp/single"),
            },
        ),
        # --- Test Sampling Sweep ---
        (
            "sweep-hp-samples --algo-and-network-config sac_mlp --env-config brax.ant --num-samples 32 --hparam-batch-size 8 --sequential False",
            "launch_sampling_sweep",
            SamplingSweepConfig,
            {
                "algo_and_network_config": "sac_mlp",
                "num_samples": 32,
                "hparam_batch_size": 8,
                "sequential": False,
            },
        ),
        # --- Test Optuna Sweep ---
        (
            "optuna-hp-search --algo-and-network-config dqn_mlp --env-config gymnax.cartpole --n-trials 50 --sampler tpe",
            "launch_optuna_sweep",
            OptunaSweepConfig,
            {"algo_and_network_config": "dqn_mlp", "n_trials": 50, "sampler": "tpe"},
        ),
        # --- Test Sample Generation ---
        (
            "generate-hp-samples --algo-and-network-config ppo_drpqc --num-samples 128 --output-file /tmp/samples.csv",
            "launch_sample_generation",
            GenerateSamplesConfig,
            {
                "algo_and_network_config": "ppo_drpqc",
                "num_samples": 128,
                "output_file": "/tmp/samples.csv",
            },
        ),
        # --- Test Benchmark Run (from config file) ---
        (
            "run-benchmark --base-config my_test_bench --output-root /tmp/bench",
            "launch_benchmark",
            BenchmarkRunConfig,
            {"base_config": "my_test_bench", "output_root": Path("/tmp/bench")},
        ),
        # --- Test Benchmark Run (from CLI args, with boolean flag) ---
        (
            "run-benchmark --benchmark-name MyBench --algos ppo_mlp sac_mlp --envs gymnax.pendulum "
            "--num-samples-per-run 16 --hparam-batch-size 8 --group-by-structural-hparams False",
            "launch_benchmark",
            BenchmarkRunConfig,
            {
                "benchmark_name": "MyBench",
                "algos": ["ppo_mlp", "sac_mlp"],
                "envs": ["gymnax.pendulum"],
                "group_by_structural_hparams": False,
            },
        ),
        # --- Test Plot Benchmark (with boolean flag) ---
        (
            "plot-benchmark --results-dir-to-plot /tmp/results --plot-mode full --include-additional-info-in-titles False",
            "launch_plot_benchmark",
            PlotBenchmarkConfig,
            {
                "results_dir_to_plot": Path("/tmp/results"),
                "include_additional_info_in_titles": False,
            },
        ),
    ],
)
def test_cli_dispatch_and_parse(
    command, expected_launcher, expected_config_type, expected_attrs, mocker, mock_launchers
):
    """
    Tests that the CLI correctly dispatches commands to their respective launch
    functions and that tyro parses all arguments into the correct config object.
    """
    run_cli_main(command, mocker)

    # 1. Check that the correct launch function was called exactly once.
    mock_launchers[expected_launcher].assert_called_once()

    # 2. Check that no other launch functions were called.
    for name, mock_fn in mock_launchers.items():
        if name != expected_launcher:
            mock_fn.assert_not_called()

    # 3. Check the type and content of the config object passed to the launcher.
    call_args, _ = mock_launchers[expected_launcher].call_args
    assert len(call_args) == 1
    config_arg = call_args[0]

    assert isinstance(config_arg, expected_config_type)
    for attr, value in expected_attrs.items():
        assert hasattr(config_arg, attr), f"Expected attribute '{attr}' not found in config of type {type(config_arg)}"
        assert getattr(config_arg, attr) == value, f"Attribute '{attr}' mismatch"
