"""Load experiment data from .npz files into DataFrames."""

import json
import logging
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd

from hyperlax.logger.return_tracker import HyperparamReturns, load_hyperparam_returns_as_named_tuples

logger = logging.getLogger(__name__)


@dataclass
class ExperimentData:
    """Container for loaded experiment data."""

    environment: str
    algorithm: str
    episodes_df: pd.DataFrame
    stats_df: pd.DataFrame
    hyperparams_df: pd.DataFrame
    metadata: dict
    run_args: dict
    source_dir: Path

    def __repr__(self):
        return (
            f"ExperimentData(env={self.environment}, algo={self.algorithm}, "
            f"n_hps={self.episodes_df['hp_id'].nunique()}, "
            f"n_timesteps={self.stats_df['timestep'].nunique()}, "
            f"n_params={self.hyperparams_df.shape[1] - 1 if not self.hyperparams_df.empty else 0})"
        )


def parse_experiment_name(dir_name: str, include_variants: bool = True) -> tuple[str, str]:
    """
    Parse directory name to extract algorithm and environment.

    Patterns parsed:
      <algo>_<env>
      <algo>_<env>_<variant>  (variant is appended to algo display name if include_variants=True)

    Examples:
        'ppo_CartPole-v1' -> ('ppo', 'CartPole-v1')
        'ppo-mlp_gymnax.pendulum_S1024' -> ('ppo-mlp (S1024)', 'gymnax.pendulum')      # if include_variants=True
        'ppo-mlp_gymnax.pendulum_S1024' -> ('ppo-mlp', 'gymnax.pendulum')               # if include_variants=False
        'dqn_Acrobot_lr3e-4' -> ('dqn (lr3e-4)', 'Acrobot')                              # if include_variants=True

    Returns:
        (algorithm_display_name, environment_name)
    """
    parts = dir_name.split('_', 2)
    if len(parts) < 2:
        raise ValueError(f"Cannot parse directory name: {dir_name}")

    algo, env = parts[0], parts[1]
    variant = parts[2] if len(parts) > 2 else None

    algo_display = f"{algo} ({variant})" if (include_variants and variant) else algo
    return algo_display, env


def convert_to_episodes_dataframe(hp_returns: list[HyperparamReturns]) -> pd.DataFrame:
    """
    Convert nested HyperparamReturns to flat DataFrame.

    Columns: hp_id, timestep, episode_idx, return, length
    """
    rows = []
    for hp_run in hp_returns:
        hp_id = hp_run.sample_id

        # Extract episode returns (dict[timestep, array])
        for timestep, returns_array in hp_run.episode_returns.items():
            returns = np.atleast_1d(returns_array)

            for episode_idx, ret in enumerate(returns):
                if not np.isnan(ret):  # Skip NaN episodes
                    rows.append({
                        'hp_id': hp_id,
                        'timestep': int(timestep),
                        'episode_idx': episode_idx,
                        'return': float(ret),
                    })

    if not rows:
        return pd.DataFrame(columns=['hp_id', 'timestep', 'episode_idx', 'return'])

    df = pd.DataFrame(rows)
    df = df.sort_values(['hp_id', 'timestep', 'episode_idx']).reset_index(drop=True)
    return df


def convert_to_stats_dataframe(hp_returns: list[HyperparamReturns]) -> pd.DataFrame:
    """
    Extract return_stats to flat DataFrame.

    Columns: hp_id, timestep, mean, std, min, max, median, q25, q75, bootstrap_ci_lower, bootstrap_ci_upper
    """
    rows = []
    for hp_run in hp_returns:
        hp_id = hp_run.sample_id

        for timestep, stats_dict in hp_run.return_stats.items():
            if not isinstance(stats_dict, dict):
                continue

            row = {
                'hp_id': hp_id,
                'timestep': int(timestep),
            }

            # Extract standard stats
            for key in ['mean', 'std', 'min', 'max', 'median', 'q25', 'q75',
                       'bootstrap_ci_lower', 'bootstrap_ci_upper']:
                row[key] = float(stats_dict.get(key, np.nan))

            rows.append(row)

    if not rows:
        return pd.DataFrame(columns=['hp_id', 'timestep', 'mean'])

    df = pd.DataFrame(rows)
    df = df.sort_values(['hp_id', 'timestep']).reset_index(drop=True)
    return df


def _normalize_param_value(value):
    """
    Convert numpy scalars/arrays to plain Python types where possible.
    Keeps sequences with >1 elements as tuples to avoid accidental coercion.
    """
    if isinstance(value, (np.generic,)):
        return value.item()
    if isinstance(value, np.ndarray):
        if value.size == 1:
            return value.reshape(()).item()
        return tuple(value.tolist())
    return value


def convert_to_hyperparams_dataframe(hp_returns: list[HyperparamReturns]) -> pd.DataFrame:
    """
    Flatten per-sample hyperparameter dictionaries into a wide DataFrame.

    Columns: hp_id, <param_path>...
    """
    rows = []
    for hp_run in hp_returns:
        params = hp_run.hyperparams or {}
        row = {'hp_id': hp_run.sample_id}
        if isinstance(params, dict):
            for key, value in params.items():
                row[str(key)] = _normalize_param_value(value)
        rows.append(row)

    if not rows:
        return pd.DataFrame(columns=['hp_id'])

    df = pd.DataFrame(rows)
    df = df.sort_values('hp_id').reset_index(drop=True)
    return df


def load_experiment_data(
    result_dir: Path,
    include_variants: bool = True,
) -> Optional[ExperimentData]:
    """
    Load experiment data from a single result directory.

    Args:
        result_dir: Directory containing 'return_group_by_hyperparams.npz'
        include_variants: Include run-specific info in algorithm name

    Returns:
        ExperimentData or None if loading fails
    """
    try:
        algo_display, env = parse_experiment_name(result_dir.name, include_variants)
    except ValueError as e:
        logger.warning(f"Skipping {result_dir.name}: {e}")
        return None

    # Find .npz files
    npz_files = list(result_dir.rglob('return_group_by_hyperparams.npz'))
    if not npz_files:
        logger.warning(f"No .npz files found in {result_dir}")
        return None

    # Load all HP returns
    all_hp_returns = []
    for npz_path in npz_files:
        try:
            hp_returns = load_hyperparam_returns_as_named_tuples(npz_path)
            all_hp_returns.extend(hp_returns)
        except Exception as e:
            logger.error(f"Failed to load {npz_path}: {e}")
            continue

    if not all_hp_returns:
        logger.warning(f"No valid data loaded from {result_dir}")
        return None

    # Convert to DataFrames
    episodes_df = convert_to_episodes_dataframe(all_hp_returns)
    stats_df = convert_to_stats_dataframe(all_hp_returns)
    hyperparams_df = convert_to_hyperparams_dataframe(all_hp_returns)

    # Load metadata (training config)
    metadata = {}
    try:
        config_path = next(result_dir.rglob('config.yaml'))
        import yaml
        with open(config_path) as f:
            config = yaml.safe_load(f)
            metadata = {
                'total_timesteps': config['training']['total_timesteps'],
                'num_evaluation': config['training']['num_evaluation'],
            }
    except (StopIteration, KeyError, FileNotFoundError):
        logger.warning(f"Could not load metadata from {result_dir}, using defaults")
        metadata = {'total_timesteps': 1_000_000, 'num_evaluation': 20}

    try:
        args_path = result_dir / "args.json"
        run_args = {}
        if args_path.exists():
            import json

            with open(args_path) as f:
                run_args = json.load(f)
    except Exception as exc:
        logger.warning("Failed to load args for %s: %s", result_dir, exc)
        run_args = {}

    return ExperimentData(
        environment=env,
        algorithm=algo_display,
        episodes_df=episodes_df,
        stats_df=stats_df,
        hyperparams_df=hyperparams_df,
        metadata=metadata,
        run_args=run_args,
        source_dir=result_dir,
    )


def load_all_experiments(
    result_dirs: list[Path],
    include_variants: bool = True,
) -> dict[str, dict[str, ExperimentData]]:
    """
    Load all experiments grouped by environment.

    Returns:
        {environment: {algorithm: ExperimentData}}
    """
    grouped = {}

    for result_dir in result_dirs:
        exp_data = load_experiment_data(result_dir, include_variants)
        if exp_data is None:
            continue

        env = exp_data.environment
        algo = exp_data.algorithm

        if env not in grouped:
            grouped[env] = {}

        grouped[env][algo] = exp_data

    return grouped
