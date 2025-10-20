from dataclasses import dataclass, field
from pathlib import Path


@dataclass
class BaseConfig:
    run_length_modifier: str | int = "default"
    """Modifier for run length (quick, default, long) or pass the timestamp directly. Affects total_timesteps etc."""

    output_root: Path | None = None
    """Base output directory for the study/sweep. Subdirectories will be created inside."""

    seed: int = 42
    """Global random seed for samplers and reproducibility."""

    log_level: str = "ERROR"
    """Logging level (DEBUG, INFO, WARNING, ERROR)."""

    save_logs: bool = True
    """If True, saves the full console output of the experiment to a file inside its output directory."""

    # This field will be populated by the launcher after directory setup.
    # output_dir: Path = field(init=False)
    output_dir: Path | None = None


@dataclass
class BaseSweepConfig(BaseConfig):
    """Common arguments for all sweep experiments."""

    algo_and_network_config: str = "ppo_mlp"
    """Name of the base algorithm recipe (e.g., dqn_mlp_base)."""

    env_config: str = "gymnax.pendulum"
    """Environment identifier (e.g., gymnax.cartpole)."""


@dataclass
class SamplingSweepConfig(BaseSweepConfig):
    """Run a hyperparameter sweep by generating samples from distributions."""

    load_hyperparams: str | None = None
    """Path to a file (CSV/JSON) containing hyperparameters to load instead of sampling."""

    num_samples: int = 8
    """Number of samples to generate if not loading from a file."""

    sampling_method: str = "qmc_sobol"
    """Sampling method ('qmc_sobol' or 'random')."""

    experiment_type: str = "independent_samples"
    """Type of sampling experiment ('sobol_matrices_A_B_AB' or 'independent_samples')."""

    omit_ab: bool = False
    """For Sobol sampling, omit the AB matrices."""

    hparam_batch_size: int = 4
    """Maximum number of hyperparameter samples per vectorized run."""

    min_hparam_batch_size: int | None = None
    """Minimum hyperparameter samples in the last sliced batch."""

    sequential: bool = False
    """If True, run each hyperparameter sample sequentially instead of in batches. Overrides hparam_batch_size."""

    group_by_structural_hparams: bool = False
    """If True, groups samples by structural (integer) hyperparameters to create homogeneous, efficient batches."""

    resume: bool = False
    """Resume from the 'latest' run in the specific output directory."""

    resume_from: str | None = None
    """Path to a specific experiment to resume from."""


@dataclass
class OptunaSweepConfig(BaseSweepConfig):
    """Run a hyperparameter optimization study using Optuna w/ three-layer process exploiting hyperlax"""

    study_name: str = "hyperlax_optuna_study"
    """Name of the Optuna study."""

    storage: str = "optuna_study.db"
    """Filename for the Optuna SQLite database. It will be saved inside the main output directory."""

    n_trials: int = 100
    """Defines the total number of experiments (trials) you want Optuna to run."""

    objective_names: list[str] = field(default_factory=lambda: ["peak_performance"])
    """List of objective metric names to optimize."""

    objective_directions: list[str] = field(default_factory=lambda: ["maximize"])
    """List of optimization directions ('maximize' or 'minimize') for each objective."""

    sampler: str = "tpe"
    """Optuna sampler to use ('tpe' or 'nsgaii' for multi-objective)."""

    optuna_study_batch_size: int = 1
    """Decides how many trials to request from the Optuna study at once. It doesn't run them itself; it just gathers a batch of work. This is the 'Super-Batch'. See run_optuna_sweep_impl fn."""

    runner_hparam_batch_size: int = 1
    """Takes the 'Super-Batch' of trials provided by the launcher and executes them. It further slices this super-batch into smaller chunks that can be run in parallel on the GPU. See run_hyperparam_batched_experiments fn."""

    resume: bool = False
    """Resume an existing study if it exists."""

    resume_from: str | None = None
    """Path to a specific experiment study to resume from."""

    save_optuna_plots: bool = True
    """Save Optuna analysis plots after the study."""


@dataclass
class BaseRunConfig(BaseConfig):
    """Common arguments for experiment runs."""

    algo_and_network_config: str = "ppo_mlp"
    """Name of the base algorithm recipe (e.g., dqn_mlp_base)."""

    env_config: str = "gymnax.pendulum"
    """Environment identifier (e.g., gymnax.cartpole)."""


@dataclass
class SingleRunConfig(BaseRunConfig):
    """Run a single experiment with default or specified hyperparameters."""

    # This command inherits all arguments from BaseRunConfig and adds no new ones.


@dataclass
class GenerateSamplesConfig(BaseConfig):
    """Generate a set of hyperparameter samples and save them to a file."""

    algo_and_network_config: str = "ppo_mlp"
    """Name of the base algorithm recipe (e.g., dqn_mlp_base)."""

    output_file: str = "hyperparam_samples.csv"
    """Path to the output CSV file to save the samples."""

    env_config: str | None = None
    """Environment identifier (e.g., gymnax.cartpole). Not required for sample generation."""

    num_samples: int = 256
    """Number of samples to generate."""

    sampling_method: str = "qmc_sobol"
    """Sampling method ('qmc_sobol' or 'random')."""

    run_length_modifier: str = "default"
    """Modifier for base hyperparameter distributions (quick, default, long)."""

    seed: int = 42
    """Global random seed for samplers."""

    log_level: str = "ERROR"
    """Logging level (DEBUG, INFO, WARNING, ERROR)."""

    save_logs: bool = False
    """If True, saves the full console output of the experiment to a file inside its output directory."""


@dataclass
class BenchmarkRunConfig(BaseConfig):
    """Run a benchmark suite from a configuration file."""

    base_config: str = "ppo_mlp_vs_drpqc"
    """Name of the benchmark config file in hyperlax/configs/benchmark/ (e.g., ppo_mlp_vs_dqn_mlp_quick)."""

    benchmark_name: str | None = None
    algos: list[str] | None = None
    envs: list[str] | None = None
    sweep_modes: list[str] | None = None
    num_samples_per_run: int | None = None
    hparam_batch_size: int | None = 8

    group_by_structural_hparams: bool = False
    """If True, groups samples by structural (integer) hyperparameters to create homogeneous, efficient batches."""

    resume: bool = False
    """Resume an existing benchmark run if it exists."""

    resume_from: str | None = None
    """Path to a specific benchmark directory to resume from."""

@dataclass
class PlotBenchmarkConfig:
    """Analyze and plot results from a benchmark run directory."""

    results_dir_to_plot: Path
    """Root directory of the benchmark results to analyze."""

    log_level: str = "ERROR"
    """Logging level for the analysis script."""

    include_additional_info_in_titles: bool = True
    """If True, adds directory's trailing info (e.g., '_N128') to algorithm names in plots."""

    top_n: int = 5
    """Top-n best found hyperparam performance for aggregation"""

    slice_metric: str = "peak_return"
    """Metric column used for slice plots (e.g., 'peak_return', 'final_return', 'score')."""

    max_workers: int = 1
    """Number of worker threads to use for analysis generation (set >1 to parallelize)."""

    combined_only: bool = False
    """If True, only plots combined single boxplot figure"""
