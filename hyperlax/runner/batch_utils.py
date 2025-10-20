import logging
import os
from collections import defaultdict
from dataclasses import replace
from typing import Any

import jax
import jax.numpy as jnp
import pandas as pd

from hyperlax.configs.main_base import BaseExperimentConfig
from hyperlax.hyperparam.base_types import HyperparamBatchGroup, flatten_tunables
from hyperlax.hyperparam.batch import HyperparamBatch

logger = logging.getLogger(__name__)


def group_samples_into_batches(
    full_batch_sample: dict[str, list[Any]],
    vectorized_keys: set,
    non_vectorized_keys: set,
    all_default_values: dict[str, Any],
    sample_id_key: str,
) -> list[HyperparamBatchGroup]:
    """Groups a flat dictionary of samples into HyperparamBatchGroup objects."""
    if not full_batch_sample:
        return []

    sampled_keys = set(full_batch_sample.keys())
    if not sampled_keys:
        return []

    batch_size = len(full_batch_sample[next(iter(sampled_keys))])
    if sample_id_key not in full_batch_sample:
        raise ValueError(f"Missing '{sample_id_key}' needed for tracking.")

    groups = defaultdict(lambda: {"vec_batches": [], "sample_ids": []})

    for idx in range(batch_size):
        current_non_vec_values = {
            key: full_batch_sample[key][idx] for key in non_vectorized_keys if key in sampled_keys
        }

        def make_hashable(val: Any) -> Any:
            if isinstance(val, list):
                return tuple(val)
            return val

        hashable_items = {k: make_hashable(v) for k, v in current_non_vec_values.items()}
        non_vec_tuple = tuple(sorted(hashable_items.items()))

        current_vec_values = {
            key: full_batch_sample[key][idx]
            for key in vectorized_keys
            if key in sampled_keys and key != sample_id_key
        }
        sample_id_val = int(full_batch_sample[sample_id_key][idx])

        groups[non_vec_tuple]["vec_batches"].append(current_vec_values)
        groups[non_vec_tuple]["sample_ids"].append(sample_id_val)

    fixed_defaults = {k: v for k, v in all_default_values.items() if k not in sampled_keys}

    return [
        HyperparamBatchGroup(
            non_vec_values=dict(non_vec_tuple),
            vec_batches=group_data["vec_batches"],
            sample_ids=group_data["sample_ids"],
            default_values=fixed_defaults,
        )
        for non_vec_tuple, group_data in groups.items()
    ]


def slice_batches(
    batch_groups: list[HyperparamBatchGroup],
    max_batch_size: int,
    min_batch_size: int | None = None,
) -> list[HyperparamBatchGroup]:
    """Slice batch groups into smaller groups based on maximum batch size.

    Args:
        batch_groups: Original list of batch groups
        max_batch_size: Maximum number of vectorized batches per group
        min_batch_size: Minimum allowed batch size (defaults to max_batch_size//2 + 1 for > 0)

    Returns:
        List of batch groups with vectorized batches split according to constraints

    Raises:
        ValueError: If max_batch_size < 1 or min_batch_size > max_batch_size

    Example:
        with a case of 4 samples and max_batch_size=3 (min_batch_size=2):
            It will create [2, 2] batches.
        with 7 samples and max_batch_size=3 (min_batch_size=2):
            It will create [3, 3, 1]. Since 1 < min_batch_size(2), redistribute.
            Total = 7. Try 2 groups: 7 // 2 = 3 rem 1. -> [4, 3]. Both >= 2. OK. -> Creates [4, 3].
            Let's trace again:
            7 // 3 = 2 rem 1. num_full_slices = 2. remainder = 1.
            remainder(1) < min_batch_size(2). Need redistribution.
            Target slices = num_full_slices = 2.
            total_slots = 7.
            slots_per_slice = 7 // 2 = 3. rem_slots = 7 % 2 = 1.
            Slice sizes: [3+1, 3] -> [4, 3]. OK.

        with 5 samples and max_batch_size=3 (min_batch_size=2):
            5 // 3 = 1 rem 2. num_full_slices = 1. remainder = 2.
            remainder(2) >= min_batch_size(2). No redistribution needed.
            Slice 1: size 3.
            Slice 2: size 2 (remainder). -> Creates [3, 2]. OK.
    """
    if max_batch_size < 1:
        raise ValueError("max_batch_size must be at least 1")

    if min_batch_size is None:
        min_batch_size = max(1, max_batch_size // 2)

    if min_batch_size > max_batch_size:
        raise ValueError(
            f"min_batch_size ({min_batch_size}) cannot be greater than max_batch_size ({max_batch_size})"
        )
    if min_batch_size < 1:
        logger.warning(f"Warning: min_batch_size ({min_batch_size}) is less than 1. Setting to 1.")
        min_batch_size = 1

    sliced_groups: list[HyperparamBatchGroup] = []

    for group in batch_groups:
        num_batches = len(group.vec_batches)
        if num_batches != len(group.sample_ids):
            raise ValueError(
                f"Data corruption: Mismatched lengths in group! vec_batches ({num_batches}) != sample_ids ({len(group.sample_ids)})"
            )
        if num_batches <= max_batch_size:
            if num_batches >= min_batch_size:
                sliced_groups.append(group)
            else:
                logger.warning(
                    f"Warning: Original group has {num_batches} batches (< min_batch_size {min_batch_size}). Keeping as is."
                )
                sliced_groups.append(group)
            continue

        num_slices = (num_batches + max_batch_size - 1) // max_batch_size
        base_slice_size = num_batches // num_slices
        remainder = num_batches % num_slices

        current_batch_slices = []
        current_id_slices = []
        start_idx = 0
        for i in range(num_slices):
            current_slice_size = base_slice_size + (1 if i < remainder else 0)
            end_idx = start_idx + current_slice_size
            current_batch_slices.append(group.vec_batches[start_idx:end_idx])
            current_id_slices.append(group.sample_ids[start_idx:end_idx])
            start_idx = end_idx

        if len(current_batch_slices[-1]) < min_batch_size and len(current_batch_slices) > 1:
            last_batch_slice = current_batch_slices.pop()
            last_id_slice = current_id_slices.pop()
            needed = min_batch_size - len(last_batch_slice)
            if len(current_batch_slices[-1]) > needed:
                moved_batch_items = current_batch_slices[-1][-needed:]
                moved_id_items = current_id_slices[-1][-needed:]
                current_batch_slices[-1] = current_batch_slices[-1][:-needed]
                current_id_slices[-1] = current_id_slices[-1][:-needed]
                current_batch_slices.append(moved_batch_items + last_batch_slice)
                current_id_slices.append(moved_id_items + last_id_slice)
            else:
                second_last_batch_slice = current_batch_slices.pop()
                second_last_id_slice = current_id_slices.pop()
                current_batch_slices.append(second_last_batch_slice + last_batch_slice)
                current_id_slices.append(second_last_id_slice + last_id_slice)

        for vec_batch_slice, sample_id_slice in zip(
            current_batch_slices, current_id_slices, strict=False
        ):
            if len(vec_batch_slice) > 0:
                sliced_groups.append(
                    replace(group, vec_batches=vec_batch_slice, sample_ids=sample_id_slice)
                )
                if len(vec_batch_slice) < min_batch_size:
                    logger.warning(
                        f"Warning: Created a slice with {len(vec_batch_slice)} batches (< min_batch_size {min_batch_size}). Check IDs: {sample_id_slice}"
                    )

    return sliced_groups


def sort_batch_groups_by_memory_impact(
    batch_groups: list[HyperparamBatchGroup],
) -> list[HyperparamBatchGroup]:
    """Sorts batch groups by estimated memory impact (highest to lowest)."""
    logger.info("\n=== Sorting Batch Groups by Memory Impact ===")
    logger.info(f"Number of batch groups to sort: {len(batch_groups)}")

    def get_memory_score(batch: dict[str, Any]) -> float:
        """Calculate memory score for a single batch configuration."""
        memory_params = {
            "algorithm.hyperparam.total_num_envs": 1.0,
            "algorithm.hyperparam.total_batch_size": 1.0,
            "algorithm.hyperparam.total_buffer_size": 1.0,
            "algorithm.hyperparam.rollout_length": 1.0,
            "algorithm.hyperparam.epochs": 0.5,
        }
        score = 1.0
        for param, weight in memory_params.items():
            if param in batch:
                value = batch[param]
                if isinstance(value, (int, float)) and value > 0:
                    score *= (float(value) + 1e-6) ** weight
        return score

    def get_group_max_memory_score(group: HyperparamBatchGroup) -> float:
        """Get maximum memory score across all batches in a group."""
        base_params = {**group.default_values, **group.non_vec_values}
        max_score = 0.0
        for batch in group.vec_batches:
            full_batch = {**base_params, **batch}
            score = get_memory_score(full_batch)
            max_score = max(max_score, score)
        return max_score

    group_scores = [(i, get_group_max_memory_score(group)) for i, group in enumerate(batch_groups)]
    sorted_indices = [i for i, score in sorted(group_scores, key=lambda x: x[1], reverse=True)]
    sorted_groups = [batch_groups[i] for i in sorted_indices]

    logger.debug("\n=== Memory Impact Sorting Results ===")
    logger.debug("Groups ordered by estimated memory impact (highest to lowest):")
    for i, (orig_idx, score) in enumerate(sorted(group_scores, key=lambda x: x[1], reverse=True)):
        logger.debug(f"  Position {i + 1}: Original group index {orig_idx} (Score: {score:.2e})")

    return sorted_groups


def build_hyperparam_batch(
    array: jnp.ndarray,
    expected_fields: tuple[str, ...],
    base_config_component: Any,
) -> HyperparamBatch:
    """
    Factory function to create a HyperparamBatch instance from array data.
    It flattens the provided component to get the correct relative keys.
    """
    logger.debug(
        f"Building hyperparam batch for component {type(base_config_component).__name__} with fields: {expected_fields} and shape: {array.shape}"
    )

    # Flatten the specific component to get its relative keys.
    # e.g., for actor_network, this gives ['pre_torso.num_layers', 'pre_torso.width', ...]
    flat_tunables = flatten_tunables(base_config_component)
    ordered_relative_keys = [path for path, spec in flat_tunables.items() if spec.is_vectorized]

    if len(ordered_relative_keys) != array.shape[1]:
        raise ValueError(
            f"Dimension mismatch: Number of vectorized tunable keys ({len(ordered_relative_keys)}) "
            f"in component '{type(base_config_component).__name__}' does not match number of columns "
            f"in hyperparameter array ({array.shape[1]}). "
            f"Keys found: {ordered_relative_keys}"
        )

    # The mapping from the simple field name (property name) to its column index
    field_name_to_index: dict[str, int] = {}
    for i, relative_key in enumerate(ordered_relative_keys):
        # The simple name is the last part of the path, e.g., 'num_layers' from 'pre_torso.num_layers'
        simple_name = relative_key.split(".")[-1]
        if simple_name in expected_fields:
            # We assume the simple name is unique within this component's vectorized HPs.
            # This holds true for both PPOHyperparams and a single network config.
            if simple_name in field_name_to_index:
                logger.warning(
                    f"Duplicate simple key '{simple_name}' found while building batch for component. "
                    f"This may lead to incorrect index mapping. Check for non-unique field names."
                )

            field_name_to_index[simple_name] = i

    logger.debug(f"field_name_to_index: {field_name_to_index}")

    # Verify that all expected fields for the batch wrapper were found and mapped.
    for field_name in expected_fields:
        if field_name not in field_name_to_index:
            raise KeyError(
                f"Could not find a matching tunable key for property '{field_name}' "
                f"in config component '{type(base_config_component).__name__}'. "
                f"Available relative keys: {ordered_relative_keys}"
            )

    return HyperparamBatch(
        data_values=array,
        field_name_to_index=field_name_to_index,
        field_names=list(expected_fields),
    )


def find_sample_id_key(base_config: BaseExperimentConfig) -> str:
    """
    Dynamically finds the full path to the 'sample_id' Tunable field.

    This is more robust than hardcoding the path, as it will adapt if the
    config structure is refactored.

    Args:
        base_config: The full experiment configuration object.

    Returns:
        The unique, dot-separated path to the 'sample_id' field.

    Raises:
        ValueError: If zero or more than one 'sample_id' fields are found.
    """
    if not hasattr(base_config, "algorithm"):
        raise ValueError(
            "Configuration object must have an 'algorithm' attribute to search for hyperparameters."
        )

    # Flatten all tunables starting from the 'algorithm' component
    flat_tunables = flatten_tunables(base_config.algorithm)

    # Find all keys that end with '.sample_id'
    # The path from flatten_tunables is relative to `config.algorithm`, so we prepend 'algorithm.'
    found_keys = [f"algorithm.{path}" for path in flat_tunables if path.endswith(".sample_id")]

    if len(found_keys) == 1:
        logger.debug(f"Dynamically found sample_id key: '{found_keys[0]}'")
        return found_keys[0]
    elif len(found_keys) == 0:
        raise ValueError(
            "Could not find a 'Tunable' field named 'sample_id' within the algorithm configuration. "
            "This field is required for tracking."
        )
    else:
        raise ValueError(
            f"Found multiple 'Tunable' fields named 'sample_id', which is ambiguous. "
            f"Please ensure it is defined only once. Keys found: {found_keys}"
        )


def _format_val(val: Any) -> str:
    """Formats a value with human-readable precision."""
    if isinstance(val, float):
        if abs(val) == 0:
            return "0.0000"
        elif abs(val) < 1e-2 or abs(val) > 1e4:
            return f"{val:.2e}"
        else:
            return f"{val:.4f}"
    return str(val)


def format_float_columns(df: pd.DataFrame) -> pd.DataFrame:
    """Creates a copy of a DataFrame with float columns formatted for display."""
    df_formatted = df.copy()
    for col_name in df_formatted.columns:
        if pd.api.types.is_float_dtype(df_formatted[col_name]):
            df_formatted[col_name] = df_formatted[col_name].apply(_format_val)
    return df_formatted


def move_column(df: pd.DataFrame, col: str, to_start: bool = True) -> pd.DataFrame:
    """Move column to the start or end of DataFrame if it exists."""
    if col in df.columns:
        cols = list(df.columns)
        cols.remove(col)
        if to_start:
            cols = [col] + cols
        else:
            cols = cols + [col]
        return df[cols]
    return df


def move_index_to_top(df: pd.DataFrame, idx_val: str) -> pd.DataFrame:
    """If DataFrame index contains idx_val, moves it to the top."""
    if idx_val in df.index:
        idxs = list(df.index)
        idxs.remove(idx_val)
        df = df.loc[[idx_val] + idxs]
    return df


def _safe_save_csv(
    df: pd.DataFrame, filename: str, save_dir: str | None, index: bool = False
) -> None:
    """Optionally saves a DataFrame to CSV if a save_dir is provided."""
    if save_dir:
        os.makedirs(save_dir, exist_ok=True)
        filepath = os.path.join(save_dir, filename)
        try:
            df.to_csv(filepath, index=index)
            logger.info(f"Saved table to {filepath}")
        except Exception as e:
            logger.warning(f"Failed to save {filename} to CSV: {e}")


def log_hyperparam_sample_preview(
    samples: dict[str, list[Any]], max_items: int = 10, save_dir: str | None = None
) -> None:
    """Print transposed preview of hyperparameter samples and optionally save."""
    try:
        if not samples:
            logger.debug("No samples to preview.")
            return
        df = pd.DataFrame(samples)
        df_preview = df.head(max_items).copy()

        # Dynamically find the sample_id column
        sample_id_col = None
        for col in df_preview.columns:
            if col.endswith(".sample_id"):
                sample_id_col = col
                break

        # Move sample_id column to front if present
        preview_front_cols = [sample_id_col] if sample_id_col else []

        # Get all other columns, sort them alphabetically
        preview_middle_cols = sorted(
            [c for c in df_preview.columns if c not in preview_front_cols]
        )

        # Reconstruct the ordered columns
        preview_ordered_cols = preview_front_cols + preview_middle_cols
        df_preview = df_preview[preview_ordered_cols]

        # Format only float columns for display
        df_formatted_preview = format_float_columns(df_preview)

        # Transpose and move 'sample_id' row to top
        df_preview_t = df_formatted_preview.T
        if sample_id_col:
            df_preview_t = move_index_to_top(df_preview_t, sample_id_col)
        # DO NOT sort_index() after move_index_to_top
        df_preview_t.columns = [f"Sample {i}" for i in range(len(df_preview_t.columns))]
        # For print_samples_preview, we want the index to be a column named "Parameter"
        df_preview_t.reset_index(inplace=True)
        df_preview_t.rename(columns={"index": "Parameter"}, inplace=True)

        logger.debug("\n" + df_preview_t.to_markdown(index=False))
        # Save original (non-formatted) transposed preview data, with sample_id top
        df_to_save = df_preview.T
        if sample_id_col:
            df_to_save = move_index_to_top(df_to_save, sample_id_col)
        _safe_save_csv(df_to_save, "hyperparams_samples_preview.csv", save_dir, index=True)
    except Exception as e:
        logger.warning(f"Failed to preview samples: {e}")


def log_batch_groups(batch_groups: list[Any], save_dir: str | None = None) -> None:
    """
    Logs a summary of hyperparameter batch groups and optionally saves them.
    If saving and only one group is provided, uses simplified filenames.
    """
    logger.debug(f"\n=== Batch Groups Summary ===\nTotal: {len(batch_groups)}\n")

    # If we are saving to a directory AND there is only one group,
    # we use simpler filenames without the 'group_XXX_' prefix.
    is_single_group_save_mode = len(batch_groups) == 1 and save_dir is not None

    for idx, group in enumerate(batch_groups, 1):
        logger.debug(f"Group {idx}:")

        if group.non_vec_values:
            lines = [f"  {k}: {v}" for k, v in sorted(group.non_vec_values.items())]
            logger.debug("\nNon-Vectorized Parameters:\n" + "\n".join(lines))
        else:
            logger.debug("\nNon-Vectorized Parameters: (none)")

        try:
            if group.vec_batches:
                df_vec = pd.DataFrame(group.vec_batches)

                # Dynamically find the sample_id column
                sample_id_col = None
                for col in df_vec.columns:
                    if col.endswith(".sample_id"):
                        sample_id_col = col
                        break

                # Move sample_id column to front if present
                vec_front_cols = [sample_id_col] if sample_id_col else []

                # Get all other columns, sort them alphabetically
                vec_middle_cols = sorted([c for c in df_vec.columns if c not in vec_front_cols])

                # Reconstruct the ordered columns
                vec_ordered_cols = vec_front_cols + vec_middle_cols
                df_vec = df_vec[vec_ordered_cols]

                # Logging to console (transposed view for better visibility)
                df_vec_fmt = format_float_columns(df_vec)
                df_vec_fmt_t = df_vec_fmt.T
                if sample_id_col:
                    df_vec_fmt_t = move_index_to_top(df_vec_fmt_t, sample_id_col)
                logger.debug(
                    f"\nVectorized Batches (n={len(group.vec_batches)}):\n"
                    + df_vec_fmt_t.to_markdown(index=True)
                )
                vec_filename = (
                    "hyperparams_vectorized.csv"
                    if is_single_group_save_mode
                    else f"hyperparams_group_{idx:03d}_vec.csv"
                )
                _safe_save_csv(df_vec, vec_filename, save_dir, index=False)
            else:
                logger.debug("\nVectorized Batches (n=0): (none)")
        except Exception as e:
            logger.warning(f"Group {idx}: Failed to log vectorized parameters: {e}")

        try:
            df_defaults = pd.DataFrame(
                sorted(group.default_values.items()), columns=["Key", "Default Value"]
            )
            if not df_defaults.empty:
                # Default values can also be mixed types.
                df_defaults["Default Value"] = df_defaults["Default Value"].astype(str)
                df_defaults_t = df_defaults.set_index("Key").T
                df_defaults_t = move_index_to_top(df_defaults_t, "sample_id")
                logger.debug("\nDefault Values:\n" + df_defaults_t.to_markdown(index=True))
                defaults_filename = (
                    "hyperparams_default.csv"
                    if is_single_group_save_mode
                    else f"hyperparams_group_{idx:03d}_defaults.csv"
                )
                # Save the transposed version as it's more readable for a few defaults
                _safe_save_csv(df_defaults_t, defaults_filename, save_dir, index=True)
            else:
                logger.debug("\nDefault Values: (none)")
        except Exception as e:
            logger.warning(f"Group {idx}: Failed to log default values: {e}")

        logger.debug("\n" + "=" * 80 + "\n")


def pretty_print_sliced_groups(
    sliced_groups: list[Any],
    group_idx: int | None = None,
    save_dir: str | None = None,
) -> None:
    """Logs a summary of sliced hyperparameter batch groups."""
    groups = [sliced_groups[group_idx]] if group_idx is not None else sliced_groups
    logger.debug(f"\n=== Sliced Batch Groups Summary ===\nTotal: {len(sliced_groups)}\n")

    for idx, group in enumerate(groups, 1):
        actual_idx = group_idx if group_idx is not None else idx
        logger.debug(f"\nSliced Group {actual_idx + 1}:\n")
        log_batch_groups([group], save_dir=save_dir)


def _sort_hyperparams_by_steps_impl(
    batch: Any,
) -> tuple[Any, jnp.ndarray, jnp.ndarray, jnp.ndarray]:
    """Core logic for sorting a hyperparameter batch by steps per update (pure function)."""
    steps_per_update_values = batch.total_num_envs * batch.rollout_length
    sort_indices = jnp.argsort(-steps_per_update_values)
    sorted_batch = batch.get_slice(sort_indices)
    sorted_steps_per_update = steps_per_update_values[sort_indices]
    return sorted_batch, sorted_steps_per_update, sort_indices, steps_per_update_values


def _log_and_save_sorted_batch(
    batch: Any,
    steps_per_update_values: jnp.ndarray,
    sort_indices: jnp.ndarray,
    save_dir: str | None = None,
) -> None:
    """Formats, logs, and optionally saves the unsorted and sorted hyperparameter batch."""
    try:
        # Convert to numpy for pandas and logging
        data_np = jax.device_get(batch.data_values)
        steps_np = jax.device_get(steps_per_update_values)

        if not hasattr(batch, "field_name_to_index"):
            logger.error(
                "Batch object is missing 'field_name_to_index'. Cannot log sorted batch table."
            )
            return

        columns = [
            k for k, v in sorted(batch.field_name_to_index.items(), key=lambda item: item[1])
        ]
        df = pd.DataFrame(data_np, columns=columns)

        # Add special columns
        df["steps_per_update"] = steps_np
        # df["original_idx"] = np.arange(len(steps_np))

        # Define special columns for ordering
        sample_id_col = "sample_id"
        end_cols_fixed_order = ["steps_per_update"]  # Add "original_idx" here if needed

        # Separate columns into categories
        front_cols = [sample_id_col] if sample_id_col in df.columns else []

        all_special_cols = set(front_cols + end_cols_fixed_order)
        unsorted_middle_cols = [c for c in df.columns if c not in all_special_cols]

        # Sort the middle columns alphabetically
        sorted_middle_cols = sorted(unsorted_middle_cols)

        # Construct the final desired column order
        ordered_cols = (
            front_cols + sorted_middle_cols + [c for c in end_cols_fixed_order if c in df.columns]
        )

        # Apply the new column order to the DataFrame
        df = df[ordered_cols]

        # Sort rows based on sort_indices and apply the new column order
        df_sorted = df.iloc[jax.device_get(sort_indices)][ordered_cols]

        # Format a copy of the head for logging, then transpose
        df_log_formatted = format_float_columns(df_sorted.head())
        df_log_transposed = df_log_formatted.T
        df_log_transposed = move_index_to_top(df_log_transposed, "sample_id")
        # DO NOT sort_index() after move_index_to_top

        # Move 'steps_per_update' to the bottom of the transposed index
        current_index = list(df_log_transposed.index)
        bottom_rows_to_move = [col for col in end_cols_fixed_order if col in current_index]
        new_index_order = [
            col for col in current_index if col not in bottom_rows_to_move
        ] + bottom_rows_to_move
        df_log_transposed = df_log_transposed.reindex(new_index_order)

        df_log_transposed.columns = [f"Sample {i}" for i in range(len(df_log_transposed.columns))]

        # The markdown table is generated with `index=True`, so the index is the first column.
        # We do not reset the index or rename the index column here for this table.

        logger.debug(
            "\nSorted Hyperparam Batch (head):\n" + df_log_transposed.to_markdown(index=True)
        )

        # Save the full (un-transposed) dataframes with original data types
        _safe_save_csv(df, "hyperparams_batch_unsorted.csv", save_dir)
        _safe_save_csv(df_sorted, "hyperparams_batch_sorted.csv", save_dir)

    except Exception as e:
        logger.warning(f"Failed to format and save hyperparam batch: {e}", exc_info=True)


def sort_hyperparams_by_steps(
    batch: Any, save_dir: str | None = None
) -> tuple[Any, jnp.ndarray, jnp.ndarray]:
    """
    Sorts a hyperparameter batch by total steps per update (descending), logs the result,
    and optionally saves it to CSV.
    """
    logger.debug(f"Starting hyperparameter sort for batch with shape: {batch.data_values.shape}")

    # 1. Core execution: pure function for sorting
    sorted_batch, sorted_steps, sort_indices, original_steps = _sort_hyperparams_by_steps_impl(
        batch
    )

    # 2. Logging and Saving: function with side-effects
    _log_and_save_sorted_batch(batch, original_steps, sort_indices, save_dir)

    return sorted_batch, sorted_steps, sort_indices
