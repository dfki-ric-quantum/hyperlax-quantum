from pathlib import Path
from types import SimpleNamespace

import pytest

from hyperlax.runner.launcher_utils import setup_output_directory


@pytest.fixture
def mock_args() -> SimpleNamespace:
    """Provides a basic mock object for command-line arguments."""
    return SimpleNamespace(
        output_root=None,
        algo_and_network_config="ppo_mlp",
        env_config="gymnax.pendulum",
        resume=False,
        resume_from=None,
    )


def test_setup_output_directory_direct_sweep_call(monkeypatch, tmp_path, mock_args):
    """
    SCENARIO: A user runs sweep-hp-samples directly.
    EXPECTED: A nested directory structure should be created, e.g., output_root/ppo_mlp/gymnax_pendulum/.
    """
    # 1. SETUP
    mock_mkdir_calls = []

    def mock_mkdir(*args, **kwargs):
        # Capture the path that mkdir was called on
        path_instance = args[0]
        mock_mkdir_calls.append(str(path_instance))

    monkeypatch.setattr(Path, "mkdir", mock_mkdir)

    mock_args.output_root = tmp_path

    # 2. ACTION
    result_path = setup_output_directory(mock_args)

    # 3. ASSERT
    expected_path = tmp_path / "ppo_mlp" / "gymnax_pendulum"
    assert result_path == expected_path, "The final path for a direct sweep call is incorrect."

    # Check that mkdir was called on the final, nested path.
    assert str(expected_path) in mock_mkdir_calls, "mkdir was not called on the expected nested path."


def test_setup_output_directory_benchmark_context(monkeypatch, tmp_path, mock_args):
    """
    SCENARIO: run-benchmark calls a sweep function, passing a pre-formatted, specific output directory.
    EXPECTED: No additional nesting should occur. The function should respect the provided path.
    """
    # 1. SETUP
    mock_mkdir_calls = []

    def mock_mkdir(*args, **kwargs):
        path_instance = args[0]
        mock_mkdir_calls.append(str(path_instance))

    monkeypatch.setattr(Path, "mkdir", mock_mkdir)

    # Simulate the pre-formatted directory name created by =launch_benchmark=
    benchmark_run_dir = tmp_path / "sac-drpqc_brax.ant_batched-N128-long"
    mock_args.output_root = benchmark_run_dir
    mock_args.algo_and_network_config = "sac_drpqc"  # This matches the directory name
    mock_args.env_config = "brax.ant"

    # 2. ACTION
    result_path = setup_output_directory(mock_args)

    # 3. ASSERT
    assert (
        result_path == benchmark_run_dir
    ), "The function should not have modified the pre-formatted path from the benchmark runner."

    # Check that mkdir was called on the original, non-nested path.
    assert str(benchmark_run_dir) in mock_mkdir_calls, "mkdir was not called on the expected root path."


def test_setup_output_directory_resume_mode(monkeypatch, tmp_path, mock_args):
    """
    SCENARIO: A sweep is resumed using --resume or --resume-from.
    EXPECTED: The provided path should be returned directly, without modification.
    """
    # 1. SETUP
    mock_mkdir_calls = []

    def mock_mkdir(*args, **kwargs):
        path_instance = args[0]
        mock_mkdir_calls.append(str(path_instance))

    monkeypatch.setattr(Path, "mkdir", mock_mkdir)

    resume_dir = tmp_path / "existing_experiment"
    mock_args.output_root = resume_dir
    mock_args.resume = True  # Enable resume mode

    # 2. ACTION
    result_path = setup_output_directory(mock_args)

    # 3. ASSERT
    assert result_path == resume_dir, "In resume mode, the output path should not be changed."

    assert str(resume_dir) in mock_mkdir_calls, "mkdir should be called on the resume directory."


def test_setup_output_directory_benchmark_with_underscored_algo(monkeypatch, tmp_path, mock_args):
    """
    SCENARIO: run-benchmark is used with an algo name containing underscores.
    EXPECTED: The check should correctly convert it to hyphens and identify the pre-formatted path.
    """
    # 1. SETUP
    mock_mkdir_calls = []
    monkeypatch.setattr(Path, "mkdir", lambda *args, **kwargs: mock_mkdir_calls.append(str(args[0])))

    # The benchmark runner will create a path with hyphens.
    benchmark_run_dir = tmp_path / "ppo-mlp-no-network-search_gymnax.pendulum_batched-N4-long"
    mock_args.output_root = benchmark_run_dir
    # The config name, however, uses underscores.
    mock_args.algo_and_network_config = "ppo_mlp_no_network_search"
    mock_args.env_config = "gymnax.pendulum"

    # 2. ACTION
    result_path = setup_output_directory(mock_args)

    # 3. ASSERT
    assert (
        result_path == benchmark_run_dir
    ), "The check should have correctly handled the underscore-to-hyphen conversion for the algo name."
    assert str(benchmark_run_dir) in mock_mkdir_calls
