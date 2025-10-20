from pathlib import Path

from hyperlax.runner.launch_args import BenchmarkRunConfig


def get_benchmark_config() -> BenchmarkRunConfig:
    """
    Defines an exampl benchmark config comparing PPO+MLP vs PPO+PQC on Pendulum
    using sequential sampling sweeps.
    """
    return BenchmarkRunConfig(
        benchmark_name="PPO_Quantum_vs_Classical",
        algos=["ppo_mlp", "ppo_drpqc"],
        envs=["gymnax.pendulum"],
        sweep_modes=["sequential"],
        num_samples_per_run=4,
        run_length_modifier="long",
        output_root=Path("results_benchmarks/"),
        log_level="CRITICAL",
    )
