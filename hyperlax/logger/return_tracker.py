"""Utilities for tracking, saving/loading and analyzing hyperparameter experiment returns."""

import logging
from collections import defaultdict
from pathlib import Path
from typing import Any, NamedTuple

import chex
import numpy as np

from hyperlax.configs.main_base import BaseExperimentConfig
from hyperlax.hyperparam.base_types import flatten_tunables
from hyperlax.utils.type_cast import cast_value_to_expected_type

logger = logging.getLogger(__name__)


class HyperparamReturns(NamedTuple):
    """Return data for a single hyperparameter configuration."""

    sample_id: int
    hyperparams: dict[str, Any]
    episode_returns: dict[int, np.ndarray]  # timestep -> returns array
    return_stats: dict[int, dict[str, float]]  # timestep -> stats from compute_return_stats


def initialize_hyperparam_returns(
    exp_config: BaseExperimentConfig,
) -> list[HyperparamReturns]:
    """
    Create initial return tracking containers for a slice of hyperparameter configurations.
    This function now handles the decoupled dictionary of hyperparameter arrays.
    """
    hp_arrays_dict = exp_config.training.hyperparam_batch_samples
    sample_ids = exp_config.training.hyperparam_batch_sample_ids
    num_configs_in_slice = len(sample_ids)

    if not sample_ids:
        return []

    # Get all tunable parameters from the config to know their paths and types
    flat_tunables = flatten_tunables(exp_config)

    # Partition all vectorized keys into their respective components ('algo', 'network_actor', etc.)
    # This must match the logic in the sampling_runner.
    key_partitions = defaultdict(list)
    for path, spec in flat_tunables.items():
        if spec.is_vectorized:
            if path.startswith("algorithm.hyperparam."):
                key_partitions["algo"].append(path)
            elif path.startswith("algorithm.network.actor_network."):
                key_partitions["network_actor"].append(path)
            elif path.startswith("algorithm.network.critic_network."):
                key_partitions["network_critic"].append(path)

    key_partitions = dict(key_partitions)
    # print("key_partitions:", key_partitions)
    # print("flat_tunables:", flat_tunables)

    return_trackers = []
    # Get the non-vectorized parameters once, as they are the same for the whole batch
    non_vectorized_params_for_group = {
        path: spec.value for path, spec in flat_tunables.items() if not spec.is_vectorized
    }

    for i in range(num_configs_in_slice):
        # Start with the non-vectorized values
        hyperparams_for_current_sample = non_vectorized_params_for_group.copy()

        # Iterate through each component ('algo', 'network_actor', etc.)
        for component_name, component_keys in key_partitions.items():
            if component_name in hp_arrays_dict and component_keys:
                component_array = hp_arrays_dict[component_name]
                # Populate the vectorized values for this component
                for col_idx, full_key in enumerate(component_keys):
                    raw_value = component_array[i][col_idx]
                    expected_type = flat_tunables[full_key].expected_type
                    # print(full_key)
                    hyperparams_for_current_sample[full_key] = cast_value_to_expected_type(
                        raw_value, expected_type
                    )

        actual_sample_id = int(sample_ids[i])
        return_trackers.append(
            HyperparamReturns(
                sample_id=actual_sample_id,
                hyperparams=hyperparams_for_current_sample,
                episode_returns={},
                return_stats={},
            )
        )

    return return_trackers


def update_hyperparam_returns(
    return_trackers: list[HyperparamReturns],
    active_indices: np.ndarray | chex.Array,  # These are the original indices of active HPs
    timesteps: list[int],  # List of environment steps per config (matching active_indices order)
    raw_episode_returns_reduced: np.ndarray,  # Shape (num_active_hps, num_total_episodes_per_hp)
    aggregated_eval_stats: dict[
        str, np.ndarray
    ],  # Contains per-HP stats (e.g., shape (num_total_hps,))
) -> list[HyperparamReturns]:
    updated_trackers = list(return_trackers)  # Make a copy of the list

    for i, active_idx_orig in enumerate(
        active_indices
    ):  # active_idx_orig is the original HP index
        timestep = timesteps[i]

        config_returns_for_hp = np.array(raw_episode_returns_reduced[i], copy=True)

        # Ensure active_idx_orig is an integer for indexing Python lists
        active_idx_int = int(active_idx_orig)

        # Get the tracker corresponding to the original HP index
        if not (0 <= active_idx_int < len(updated_trackers)):
            logger.warning(
                f"Original active_idx {active_idx_int} is out of bounds for return_trackers (len {len(updated_trackers)}). Skipping."
            )
            continue
        tracker = updated_trackers[active_idx_int]

        new_episode_returns_dict = dict(tracker.episode_returns)
        new_episode_returns_dict[timestep] = config_returns_for_hp

        new_return_stats_dict = dict(tracker.return_stats)
        current_hp_stats = {}

        for stat_key, stat_value_array in aggregated_eval_stats.items():
            if stat_key.startswith("episode_return_"):
                # Ensure stat_value_array covers all original HPs and active_idx_int is a valid index for it
                if not (0 <= active_idx_int < stat_value_array.shape[0]):
                    logger.warning(
                        f"Original active_idx {active_idx_int} is out of bounds for stat_value_array '{stat_key}' (shape {stat_value_array.shape}). Assigning NaN."
                    )
                    current_hp_stats[stat_key.replace("episode_return_", "")] = float(np.nan)
                    continue

                # CORRECTED LINE: Use the original HP index (active_idx_int) to get its specific stat
                val_for_hp = stat_value_array[active_idx_int]
                current_hp_stats[stat_key.replace("episode_return_", "")] = float(val_for_hp)
            # You might optionally add episode_length stats here if HyperparamReturns.return_stats should contain them
            # elif stat_key.startswith("episode_length_"):
            # ... similar logic ...

        new_return_stats_dict[timestep] = current_hp_stats

        updated_trackers[active_idx_int] = tracker._replace(
            episode_returns=new_episode_returns_dict, return_stats=new_return_stats_dict
        )

    return updated_trackers


def save_hyperparam_returns(
    return_trackers: list[HyperparamReturns],
    save_path: Path | str,
):
    save_path = Path(save_path) if type(save_path) == str else save_path
    return_group_by_hyperparams = []
    for tracker in return_trackers:
        group_dict = {
            "sample_id": tracker.sample_id,
            "hyperparam": tracker.hyperparams,
            "return": tracker.episode_returns,
            "return_stats": tracker.return_stats,
        }
        return_group_by_hyperparams.append(group_dict)
    # jnp.savez(save_path, return_group_by_hyperparams=return_group_by_hyperparams)
    np.savez(save_path, return_group_by_hyperparams=return_group_by_hyperparams)
    return return_group_by_hyperparams


def load_hyperparam_returns_as_named_tuples(load_path: Path) -> list[HyperparamReturns]:
    load_path = Path(load_path) if type(load_path) == str else load_path
    data = np.load(load_path, allow_pickle=True)
    return_group = data["return_group_by_hyperparams"]

    def safe_extract(obj):
        """Safely extract value from numpy array or return as-is if already a Python object."""
        if hasattr(obj, "item") and hasattr(obj, "shape") and obj.shape == ():
            return obj.item()
        return obj

    return [
        HyperparamReturns(
            sample_id=group["sample_id"],
            hyperparams=safe_extract(group["hyperparam"]),
            episode_returns=safe_extract(group["return"]),
            return_stats=safe_extract(group["return_stats"]),
        )
        for group in return_group
    ]


def load_hyperparam_returns_as_raw(load_path: Path) -> list[dict]:
    data = np.load(load_path, allow_pickle=True)
    return data["return_group_by_hyperparams"]
