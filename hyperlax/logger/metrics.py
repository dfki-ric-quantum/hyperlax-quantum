import logging
from pathlib import Path
from typing import Any

import chex
import jax
import jax.numpy as jnp
import numpy as np
import pandas as pd
from colorama import Fore, Style
from tabulate import tabulate

from hyperlax.base_types import HPRuntimeState
from hyperlax.configs.main_base import BaseExperimentConfig as ExperimentConfig
from hyperlax.logger.return_tracker import HyperparamReturns

logger = logging.getLogger(__name__)


def is_scalar(x):
    """Return True if x is a scalar (shape = ()), False otherwise."""
    return hasattr(x, "shape") and x.shape == ()



def _compute_bootstrap_cis_for_hp_slice(
    returns_slice: jnp.ndarray, key: chex.PRNGKey
) -> tuple[jnp.ndarray, jnp.ndarray]:
    """
    Computes 95% bootstrap confidence intervals for a single HP's episode returns.
    returns_slice has shape (num_episodes,).
    """
    n_bootstrap = 1000
    total_samples = returns_slice.shape[0]  # This array size is static for each vmapped slice

    # Calculate number of valid (non-NaN) samples. This is a concrete integer value within JIT.
    n_valid = jnp.sum(~jnp.isnan(returns_slice))

    # --- Branches for jax.lax.cond ---
    # Both branches must have the same signature: (key_arg, returns_slice_arg) -> (lower_ci, upper_ci)

    # Case 1: Not enough valid data to compute a meaningful CI (n_valid < 2)
    # For 0 or 1 valid samples, a confidence interval is not meaningful.
    # We return NaN for lower and upper bounds in this case.
    def handle_insufficient_data(key_arg: chex.PRNGKey, returns_slice_arg: jnp.ndarray):
        return jnp.array(jnp.nan, dtype=jnp.float32), jnp.array(jnp.nan, dtype=jnp.float32)

    # Case 2: Sufficient valid data to perform bootstrapping (n_valid >= 2)
    def perform_bootstrapping(key_arg: chex.PRNGKey, returns_slice_arg: jnp.ndarray):
        bootstrap_keys = jax.random.split(key_arg, n_bootstrap)

        # This inner function samples indices from the original `returns_slice_arg` range
        # and then computes the mean, robustly handling NaNs in the resampled data.
        def bootstrap_sample_mean_single(inner_key, data_slice, num_original_samples):
            # Sample indices with replacement from the original range [0, num_original_samples-1]
            indices = jax.random.randint(
                inner_key, (num_original_samples,), 0, num_original_samples
            )
            resampled_data = data_slice[indices]
            # Use jnp.nanmean to correctly handle any NaNs that might be present in the resampled data
            # (e.g., if the original data had NaNs, and we resampled those NaN indices).
            return jnp.nanmean(resampled_data)

        # Vmap `bootstrap_sample_mean_single` over the `n_bootstrap` keys.
        # `returns_slice_arg` and `total_samples` are passed as static arguments (None in in_axes).
        bootstrap_means = jax.vmap(bootstrap_sample_mean_single, in_axes=(0, None, None))(
            bootstrap_keys, returns_slice_arg, total_samples
        )

        # Calculate confidence intervals from the sorted bootstrap means.
        sorted_means = jnp.sort(bootstrap_means)
        lower_idx = int(n_bootstrap * 0.025)
        upper_idx = int(n_bootstrap * 0.975)
        return sorted_means[lower_idx].astype(jnp.float32), sorted_means[upper_idx].astype(
            jnp.float32
        )

    # Use jax.lax.cond to select the appropriate branch based on `n_valid`.
    # Both `handle_insufficient_data` and `perform_bootstrapping` functions receive
    # `key` and `returns_slice` as arguments.
    ci_lower, ci_upper = jax.lax.cond(
        n_valid < 2,  # Condition: True if less than 2 valid points (0 or 1)
        handle_insufficient_data,  # True branch function
        perform_bootstrapping,  # False branch function
        key,
        returns_slice,  # Operands passed to both functions
    )

    return ci_lower, ci_upper


# Vmap the bootstrap CI function over the HP dimension
vmap_compute_bootstrap_cis = jax.vmap(_compute_bootstrap_cis_for_hp_slice, in_axes=(0, 0))


def aggregate_metrics_per_hyperparam(
    metrics_dict: dict,
    num_hyperparams: int,
    prefix: str = "",
    skip_metric_key: tuple = (),
    n_bootstrap: int = 1000,
) -> dict:
    """
    Aggregate metrics per hyperparameter configuration.
    Keeps: mean, std, min, max, count, median, q25, q75, iqr, peak (for episodes),
           and bootstrap CIs for episode arrays.
    Drops: cv, skewness, sem, parametric CI95, MAD, bootstrap_samples bookkeeping.
    """
    aggregated = {}

    bootstrap_base_key = jax.random.PRNGKey(42)
    bootstrap_keys_per_hp = jax.random.split(bootstrap_base_key, num_hyperparams)

    for key, value in metrics_dict.items():
        if key in skip_metric_key:
            continue
        if not isinstance(value, jnp.ndarray):
            try:
                value = jnp.asarray(value)
            except Exception as e:
                logger.warning(
                    f"Could not convert metric '{prefix}{key}' to JAX array: {e}. Skipping."
                )
                aggregated[f"{prefix}{key}"] = value
                continue

        base_key = f"{prefix}{key}"

        # Special handling for episode metrics
        if key in ("episode_return", "episode_length"):
            arr = value
            if arr.ndim <= 1:
                aggregated[base_key] = arr
                continue

            axes_to_aggregate_for_stats = tuple(range(1, arr.ndim))
            count_per_hp = jnp.sum(~jnp.isnan(arr), axis=axes_to_aggregate_for_stats)

            arr_mean = jnp.nanmean(arr, axis=axes_to_aggregate_for_stats)
            arr_std = jnp.nanstd(arr, axis=axes_to_aggregate_for_stats)
            arr_min = jnp.nanmin(arr, axis=axes_to_aggregate_for_stats)
            arr_max = jnp.nanmax(arr, axis=axes_to_aggregate_for_stats)

            arr_median = jnp.nanmedian(arr, axis=axes_to_aggregate_for_stats)
            arr_q25 = jnp.nanpercentile(arr, 25, axis=axes_to_aggregate_for_stats)
            arr_q75 = jnp.nanpercentile(arr, 75, axis=axes_to_aggregate_for_stats)
            arr_iqr = arr_q75 - arr_q25
            arr_peak_metric = jnp.nanmax(arr, axis=axes_to_aggregate_for_stats)

            # Bootstrap CIs per-HP over episodes
            bootstrap_ci_lower, bootstrap_ci_upper = vmap_compute_bootstrap_cis(
                arr, bootstrap_keys_per_hp
            )

            aggregated[f"{base_key}_mean"] = arr_mean
            aggregated[f"{base_key}_std"] = arr_std
            aggregated[f"{base_key}_min"] = arr_min
            aggregated[f"{base_key}_max"] = arr_max
            aggregated[f"{base_key}_count"] = count_per_hp

            aggregated[f"{base_key}_median"] = arr_median
            aggregated[f"{base_key}_q25"] = arr_q25
            aggregated[f"{base_key}_q75"] = arr_q75
            aggregated[f"{base_key}_iqr"] = arr_iqr
            aggregated[f"{base_key}_peak"] = arr_peak_metric

            aggregated[f"{base_key}_bootstrap_ci_lower"] = bootstrap_ci_lower
            aggregated[f"{base_key}_bootstrap_ci_upper"] = bootstrap_ci_upper
            continue

        # General metrics (non-episode)
        if value.ndim == 0:
            aggregated[base_key] = value
            continue
        if value.ndim == 1 and value.shape[0] == num_hyperparams:
            aggregated[base_key] = value
            continue

        if value.ndim > 1 and value.shape[0] == num_hyperparams:
            axes_to_aggregate = tuple(range(1, value.ndim))
            arr = value
            count_per_hp = jnp.sum(~jnp.isnan(arr), axis=axes_to_aggregate)

            arr_mean = jnp.nanmean(arr, axis=axes_to_aggregate)
            arr_std = jnp.nanstd(arr, axis=axes_to_aggregate)
            arr_min = jnp.nanmin(arr, axis=axes_to_aggregate)
            arr_max = jnp.nanmax(arr, axis=axes_to_aggregate)

            aggregated[f"{base_key}_mean"] = arr_mean
            aggregated[f"{base_key}_std"] = arr_std
            aggregated[f"{base_key}_min"] = arr_min
            aggregated[f"{base_key}_max"] = arr_max
            aggregated[f"{base_key}_count"] = count_per_hp
        else:
            logger.warning(
                f"Metric '{key}' has shape {value.shape}, but expected first dimension "
                f"to be num_hyperparams ({num_hyperparams}) or scalar. Skipping aggregation."
            )
            aggregated[base_key] = value

    return aggregated


def save_metrics(
    base_exp_path: str | Path,
    train_metrics_agg: dict[str, Any],
    eval_metrics_agg: dict[str, Any],
    eval_step: int,
) -> None:
    """
    Save aggregated training and evaluation metrics for all HPs to an NPZ file.

    Args:
        base_exp_path: Base experiment path for saving metrics.
        train_metrics_agg: Dictionary of fully aggregated training metrics.
        eval_metrics_agg: Dictionary of fully aggregated evaluation metrics.
        eval_step: Current evaluation step number.
    """
    base_path = Path(base_exp_path) / "metrics"
    base_path.mkdir(parents=True, exist_ok=True)  # Ensure directory exists

    # Create filename for this step
    step_file = base_path / f"eval_step_{eval_step}.npz"  # Naming based on eval step

    # Combine metrics for saving, ensuring numpy conversion for savez
    save_dict = {}

    for category_name, metrics in [
        ("train", train_metrics_agg),
        ("eval", eval_metrics_agg),
    ]:
        for metric_name, value in metrics.items():
            # Aggregated metrics are likely already numpy/jax scalars or arrays
            # Convert JAX arrays to NumPy arrays for saving compatibility
            if isinstance(value, jnp.ndarray):
                key = f"{category_name}_{metric_name}"  # Original key had prefix already
                # Use np.asarray to handle device arrays
                save_dict[key] = np.asarray(value)
            elif isinstance(value, (int, float, bool, np.number, np.bool_)):
                key = f"{category_name}_{metric_name}"
                save_dict[key] = np.array(value)  # Store scalars as 0-dim arrays
            else:
                # Handle potential non-numeric types if necessary (e.g., strings in metadata)
                key = f"{category_name}_{metric_name}"
                try:
                    # Try converting to a numpy array if possible
                    save_dict[key] = np.asarray(value)
                except Exception:
                    logger.warning(
                        f"Could not convert metric '{key}' "
                        f"of type {type(value)} to NumPy array for saving. Skipping."
                    )

    # Save using numpy's savez_compressed
    try:
        np.savez_compressed(step_file, **save_dict)
    except Exception as e:
        logger.error(f"Error saving metrics to {step_file}: {e}")
        # Consider saving problematic keys individually or logging more details

    # Optionally, save a metadata file for easy access to step information
    metadata_file = base_path / "steps_info.txt"
    try:
        with metadata_file.open("a") as f:
            f.write(f"Eval Step {eval_step} saved at: {step_file}\n")
    except Exception as e:
        logger.error(f"Error writing to metadata file {metadata_file}: {e}")

    # logger.info(f"Metrics saved for eval step {eval_step} at: {step_file.absolute()}")


def load_metrics(config: ExperimentConfig, eval_step: int | None = None) -> dict:
    """
    Load metrics from NPZ files.

    If eval_step is provided, loads metrics for that specific evaluation step.
    If eval_step is None, loads metrics for all available evaluation steps.

    Args:
        config: Experiment configuration.
        eval_step: The specific evaluation step to load, or None to load all.

    Returns:
        If eval_step is specified: A dictionary {'train': {...}, 'eval': {...}} for that step.
        If eval_step is None: A dictionary {step1: {'train': {...}, 'eval': {...}}, step2: ...}.
    """
    base_path = Path(config.logger.base_exp_path) / "metrics"
    logger.info(f"Loading metrics from base path: {base_path.absolute()}")

    if not base_path.exists():
        logger.error(f"Error: Metrics directory not found: {base_path}")
        return {}

    if eval_step is not None:
        # Load specific step
        step_file = base_path / f"eval_step_{eval_step}.npz"
        if not step_file.exists():
            logger.error(
                f"Error: No metrics file found for evaluation step {eval_step} at {step_file}"
            )
            return {}  # Return empty dict instead of raising FileNotFoundError

        logger.info(f"Loading metrics from: {step_file.absolute()}")
        try:
            with np.load(
                step_file, allow_pickle=False
            ) as data:  # Set allow_pickle=False for security
                metrics = {"train": {}, "eval": {}}
                for key in data.files:
                    # Split key into category (train/eval) and the rest (metric_name)
                    parts = key.split("_", 1)
                    if len(parts) == 2 and parts[0] in metrics:
                        category, metric_name = parts
                        metrics[category][metric_name] = data[key]
                    else:
                        # Handle keys that don't fit the pattern if necessary
                        logger.warning(f"Skipping unexpected key '{key}' in {step_file}")
            return metrics
        except Exception as e:
            logger.error(f"Error loading metrics file {step_file}: {e}")
            return {}

    else:
        # Load all steps
        logger.info("Loading all available evaluation step files...")
        all_metrics = {}
        # Ensure correct numerical sorting of steps
        step_files = sorted(
            base_path.glob("eval_step_*.npz"), key=lambda f: int(f.stem.split("_")[-1])
        )

        if not step_files:
            logger.warning(f"No evaluation step files found in {base_path}")
            return {}

        for step_file in step_files:
            try:
                step = int(step_file.stem.split("_")[-1])
                # logger.info(f"Loading step {step} from: {step_file.absolute()}") # Can be verbose
                step_metrics = load_metrics(config, step)
                if step_metrics:  # Only add if loading was successful
                    all_metrics[step] = step_metrics
            except (ValueError, IndexError):
                logger.warning(
                    f"Warning: Could not parse step number from filename {step_file.name}. Skipping."
                )
                continue  # Skip files with unexpected naming

        logger.info(f"Loaded metrics for {len(all_metrics)} evaluation steps.")
        return all_metrics


def _is_pytree_empty(pytree: Any | None) -> bool:
    """Checks if a PyTree is effectively empty (None or no leaves)."""
    if pytree is None:
        return True
    leaves, _ = jax.tree_util.tree_flatten(pytree)
    return not bool(leaves)


def _get_nan_like_tree(tree_structure: Any, num_hyperparams: int) -> Any:
    """
    Creates a PyTree with the same structure as tree_structure,
    but with a new leading dimension of size num_hyperparams, filled with NaNs
    for float types, 0 for int types, and False for bool types.

    Args:
        tree_structure: An example PyTree (e.g., metrics from one config).
        num_hyperparams: The desired size of the new leading dimension.

    Returns:
        A new PyTree with the expanded leading dimension, filled with defaults.
    """

    def _build_placeholder_leaf(leaf):
        if not hasattr(leaf, "shape") or not hasattr(leaf, "dtype"):
            # Handle non-array leaves if necessary, maybe replicate?
            # For now, assume leaves are arrays or scalars convertible to arrays
            leaf = jnp.asarray(leaf)

        original_shape = leaf.shape
        dtype = leaf.dtype
        new_shape = (num_hyperparams,) + original_shape

        if jnp.issubdtype(dtype, jnp.floating):
            return jnp.full(new_shape, jnp.nan, dtype=dtype)
        elif jnp.issubdtype(dtype, jnp.integer):
            return jnp.full(new_shape, 0, dtype=dtype)
        elif jnp.issubdtype(dtype, jnp.bool_):
            return jnp.full(new_shape, False, dtype=dtype)
        else:
            # Fallback for other types (e.g., objects), might need specific handling
            # Or raise an error if unsupported types are encountered
            try:
                # Attempt to create NaNs for unsupported float-like types
                return jnp.full(new_shape, jnp.nan, dtype=dtype)
            except (TypeError, ValueError):
                raise TypeError(f"Unsupported dtype {dtype} for metric placeholder creation.")

    return jax.tree_util.tree_map(_build_placeholder_leaf, tree_structure)


def _construct_full_metrics_tree(
    active_metrics_tree: Any,
    active_hyp_indices: jnp.ndarray,
    num_hyperparams: int,
    last_full_metrics_tree: Any | None = None,
    fill_completed_with_nan: bool = False,
) -> Any:
    num_active_configs = len(active_hyp_indices)

    # 1. Determine the reference structure for the PyTree.
    reference_leaf_structure = None
    if not _is_pytree_empty(active_metrics_tree):
        _temp_active_processed_for_struct = jax.tree_util.tree_map(
            lambda leaf: _ensure_leading_dim(leaf, max(1, num_active_configs)),
            active_metrics_tree,
        )
        if num_active_configs == 0:
            _temp_active_processed_for_struct = jax.tree_util.tree_map(
                lambda x: x[:0], _temp_active_processed_for_struct
            )
        reference_leaf_structure = jax.tree_util.tree_map(
            lambda x: x[0] if x.shape[0] > 0 else x,  # Handle (0, ...) shape correctly
            _temp_active_processed_for_struct,
        )
    elif not fill_completed_with_nan and not _is_pytree_empty(last_full_metrics_tree):
        reference_leaf_structure = jax.tree_util.tree_map(
            lambda x: (
                x[0] if x.shape[0] > 0 else x
            ),  # Handle (0, ...) shape correctly if last_full was empty before
            last_full_metrics_tree,
        )

    if _is_pytree_empty(reference_leaf_structure):
        logger.debug(
            f"[LSM_CONSTRUCT] No valid reference structure found. Active tree empty: {_is_pytree_empty(active_metrics_tree)}. "
            f"Fill NaN: {fill_completed_with_nan}. Last full tree empty: {_is_pytree_empty(last_full_metrics_tree)}. "
            f"Returning empty dict."
        )
        return {}

    # 2. Create the full placeholder (NaN-filled tree for all HPs)
    full_metrics_placeholder = _get_nan_like_tree(reference_leaf_structure, num_hyperparams)

    # 3. Prepare `active_metrics_processed` for the tree_map.
    if not _is_pytree_empty(active_metrics_tree):
        active_metrics_processed = jax.tree_util.tree_map(
            lambda leaf: _ensure_leading_dim(leaf, num_active_configs),
            active_metrics_tree,
        )
    else:
        active_metrics_processed = _get_nan_like_tree(reference_leaf_structure, num_active_configs)

    # 4. Prepare `last_tree_for_map`.
    if fill_completed_with_nan or _is_pytree_empty(last_full_metrics_tree):
        last_tree_for_map = full_metrics_placeholder
    else:
        if jax.tree_util.tree_structure(last_full_metrics_tree) != jax.tree_util.tree_structure(
            full_metrics_placeholder
        ):
            logger.warning(
                "[LSM_CONSTRUCT] Structure mismatch between last_full_metrics_tree and current reference structure. "
                "Using NaN-fill for non-active HPs. This might indicate an evolving metric set."
            )
            last_tree_for_map = full_metrics_placeholder
        else:
            last_tree_for_map = last_full_metrics_tree

    active_idx_map = {int(orig_idx): i for i, orig_idx in enumerate(active_hyp_indices)}

    def _update_leaf_simple(placeholder_leaf, processed_active_leaf, last_leaf):
        updated_leaf = placeholder_leaf

        for orig_idx in range(num_hyperparams):
            if orig_idx in active_idx_map:
                active_idx = active_idx_map[orig_idx]
                if active_idx < processed_active_leaf.shape[0]:  # Ensure index is valid
                    updated_leaf = updated_leaf.at[orig_idx].set(processed_active_leaf[active_idx])
            elif not fill_completed_with_nan and not _is_pytree_empty(
                last_full_metrics_tree
            ):  # Check if last_full_metrics_tree was provided
                if orig_idx < last_leaf.shape[0]:  # Ensure index is valid
                    updated_leaf = updated_leaf.at[orig_idx].set(last_leaf[orig_idx])
        return updated_leaf

    final_full_metrics = jax.tree_util.tree_map(
        _update_leaf_simple,
        full_metrics_placeholder,
        active_metrics_processed,
        last_tree_for_map,
    )
    return final_full_metrics


from hyperlax.layout.axes import DistributionStrategy


def reduce_metrics_over_batching_axes(
    metrics_tree: dict[str, jnp.ndarray],
    strategy: DistributionStrategy,
    metrics_category: str,  # "train" or "eval"
) -> dict[str, jnp.ndarray]:
    """
    Reduces (e.g., by averaging) metrics over batching dimensions
    (seed, device, update_batch, epoch, minibatch) to leave only
    the hyperparam dimension and intrinsic metric dimensions.

    Args:
        metrics_tree: A PyTree of metrics, where leaves have shape
                      (Axis0_dim, Axis1_dim, ..., OriginalMetricDims...).
        strategy: The DistributionStrategy used for vmap/pmap.
        metrics_category: "train" or "eval" to apply specific aggregation rules.

    Returns:
        A new PyTree where each leaf has shape (HP_dim, ReducedMetricDims...).
    """
    reduced_metrics = {}

    # Pre-calculate positions of all axes defined in the strategy
    axis_name_to_pos = {axis_spec.name: axis_spec.in_axes for axis_spec in strategy.axes}
    hp_axis_pos_in_initial_value = axis_name_to_pos.get("hyperparam")

    if hp_axis_pos_in_initial_value is None:
        raise ValueError(
            "Hyperparam axis not found in the provided strategy. Cannot reduce metrics."
        )

    for key, value in metrics_tree.items():
        if not isinstance(value, jnp.ndarray):
            logger.warning(
                f"Skipping non-JAX array metric '{key}' during reduction: {type(value)}"
            )
            reduced_metrics[key] = value
            continue

        # logger.debug(f"DEBUG_REDUCE: Processing metric '{key}', initial shape: {value.shape}, HP axis original pos: {hp_axis_pos_in_initial_value}")

        if metrics_category == "eval" and key in ("episode_return", "episode_length"):
            # Special handling for episode metrics: we want to flatten all
            # non-hyperparam batching dimensions (Seed, Device, internal eval_batch, RewardDim)
            # into a single dimension after the HP dimension.
            # We explicitly *do not* use jnp.mean over these dimensions, as we want to preserve individual episodes.

            # logger.debug(f"DEBUG_REDUCE: Metric '{key}' (eval episode) - special flattening.")

            # Step 1: Move HP axis to position 0 if it's not already.
            if hp_axis_pos_in_initial_value != 0:
                # Create a permutation that moves hp_axis_pos_in_initial_value to 0
                # and shifts other axes accordingly.
                permutation = [hp_axis_pos_in_initial_value] + [
                    i for i in range(value.ndim) if i != hp_axis_pos_in_initial_value
                ]
                current_reduced_value = jnp.transpose(value, axes=permutation)
                # logger.debug(f"DEBUG_REDUCE: Metric '{key}' moved HP axis to 0, shape: {current_reduced_value.shape}")
            else:
                # HP axis is already at position 0.
                current_reduced_value = value
                # logger.debug(f"DEBUG_REDUCE: Metric '{key}' HP axis already at 0, shape: {current_reduced_value.shape}")

            # Step 2: Flatten all dimensions *after* the HP dimension (which is now at axis 0)
            # into a single dimension. This collects all individual episode results.
            if current_reduced_value.ndim > 1:
                current_reduced_value = current_reduced_value.reshape(
                    current_reduced_value.shape[0], -1
                )
            # If ndim <= 1, it means it's already (HP,) or a scalar, no further flattening needed.

            # logger.debug(f"DEBUG_REDUCE: Metric '{key}' (eval episode) after flattening, final shape: {current_reduced_value.shape}")
            reduced_metrics[key] = current_reduced_value
            continue  # Skip other general reduction logic below

        # --- General reduction logic for other metrics (train or non-episode eval) ---
        # This part should *still* perform a mean over the identified axes, as losses etc. are inherently means/sums.

        axes_to_reduce_list = []
        for axis_spec in strategy.axes:
            if axis_spec.name != "hyperparam" and axis_spec.in_axes < value.ndim:
                axes_to_reduce_list.append(axis_spec.in_axes)

        if (
            metrics_category == "train"
        ):  # For train metrics, also reduce any dimensions beyond the strategy axes
            max_strategy_in_axis = max((axis.in_axes for axis in strategy.axes), default=-1)
            for i in range(max_strategy_in_axis + 1, value.ndim):
                axes_to_reduce_list.append(i)

        axes_to_reduce_tuple = tuple(sorted(axes_to_reduce_list))
        # logger.debug(f"DEBUG_REDUCE: Metric '{key}' (general), Axes to reduce: {axes_to_reduce_tuple}")

        current_reduced_value = value
        if axes_to_reduce_tuple:
            current_reduced_value = jnp.mean(value, axis=axes_to_reduce_tuple)
            # logger.debug(f"DEBUG_REDUCE: Metric '{key}' after general mean, shape: {current_reduced_value.shape}")

        # Move HP axis to 0 if it's not already.
        # This part must be careful if the original HP axis was included in the reduction.
        # But for general metrics, we ensured HP is *not* in axes_to_reduce_tuple.
        original_axes_kept = sorted(list(set(range(value.ndim)) - set(axes_to_reduce_tuple)))
        try:
            new_hp_axis_pos_in_reduced = original_axes_kept.index(hp_axis_pos_in_initial_value)
        except ValueError:
            logger.error(
                f"ERROR: HP axis {hp_axis_pos_in_initial_value} was unexpectedly reduced for metric '{key}'! Cannot determine new HP axis position. This should not happen for non-episode metrics."
            )
            reduced_metrics[key] = value  # Fallback
            continue

        if new_hp_axis_pos_in_reduced != 0:
            current_reduced_value = jnp.moveaxis(
                current_reduced_value, new_hp_axis_pos_in_reduced, 0
            )
            # logger.debug(f"DEBUG_REDUCE: Metric '{key}' moved HP axis to 0 for general metric, shape: {current_reduced_value.shape}")

        reduced_metrics[key] = current_reduced_value

    return reduced_metrics


def _ensure_leading_dim(leaf: Any, expected_dim_size: int) -> jnp.ndarray:
    """
    Ensures a leaf is a JAX array with a leading dimension of expected_dim_size.
    If the leaf is scalar, it broadcasts it.
    This function expects the input `leaf` to *already* be reduced to `(expected_dim_size, ...)`
    or a scalar `()`.
    """
    if not isinstance(leaf, jnp.ndarray):
        leaf = jnp.asarray(leaf)  # Convert Python scalars/lists to JAX array

    if leaf.ndim == 0:  # Scalar
        return jnp.broadcast_to(leaf, (expected_dim_size,))
    elif leaf.ndim > 0 and leaf.shape[0] == expected_dim_size:
        return leaf  # Already has the correct leading dimension
    else:
        # This branch implies an error in the reduction step before _construct_full_metrics_tree.
        # Or, it's a metric that shouldn't have been reduced to HP-first shape but is passed anyway.
        # Given the previous context, `_construct_full_metrics_tree` should only receive HP-batched or scalar values.

        # If it's a scalar value (e.g., total_timesteps), we can broadcast it.
        # Check if it represents a single value that needs to be replicated.
        if leaf.size == 1:
            try:
                scalar_val = leaf.item()  # Get Python scalar
                return jnp.broadcast_to(jnp.asarray(scalar_val), (expected_dim_size,))
            except ValueError:
                pass  # Fall through to error if item() fails

        # If execution reaches here, it's an unhandled shape.
        raise ValueError(
            f"Cannot handle array leaf with shape {leaf.shape}. "
            f"Expected leading dimension {expected_dim_size} or a scalar. "
            f"This suggests a problem in the preceding metric reduction step or an unexpected global scalar."
        )


def log_and_save_aggregated_metrics(
    base_exp_path: str | Path,
    aggregate_metrics_flag: bool,
    active_train_metrics: dict,
    active_eval_metrics: dict,
    active_hyp_indices: jnp.ndarray,
    num_hyperparams: int,
    last_full_eval_metrics: dict[str, Any] | None,
    eval_step: int,
    train_strategy: DistributionStrategy,
    eval_strategy: DistributionStrategy,
) -> tuple[dict[str, Any], dict[str, Any]] | None:
    """
    Aggregates training and evaluation metrics for active hyperparameter configurations,
    constructs full metric trees, and saves them.

    Args:
        base_exp_path: Base experiment path for saving metrics.
        aggregate_metrics_flag: Whether to aggregate and save metrics.
        active_train_metrics: Dict of training metrics from active HPs, with all batching dims.
        active_eval_metrics: Dict of evaluation metrics from active HPs, with all batching dims.
        active_hyp_indices: Indices of the currently active hyperparameter configurations.
        num_hyperparams: Total number of hyperparameter configurations.
        last_full_eval_metrics: The full evaluation metrics tree from the previous step.
        eval_step: The current evaluation step.
        train_strategy: The DistributionStrategy used for training.
        eval_strategy: The DistributionStrategy used for evaluation.

    Returns:
        The updated full evaluation metrics tree for the next iteration.
    """

    logger.debug(f"Called log_and_save_aggregated_metrics for eval_step: {eval_step}")
    # logger.debug(f"aggregate_metrics_flag: {aggregate_metrics_flag}")
    # logger.debug(f"active_train_metrics shapes: {get_pytree_shapes(active_train_metrics)}")
    # logger.debug(f"active_eval_metrics shapes: {get_pytree_shapes(active_eval_metrics)}")

    if not aggregate_metrics_flag:
        logger.info("Skipping metrics aggregation and saving as aggregate_metrics_flag is False.")
        return last_full_eval_metrics, {}  # Return empty eval_agg if skipping

    # --- Reduce active_train_metrics over its batching dimensions ---
    # This will output metrics with shape (HP, OriginalMetricDims...) or (HP,)
    active_train_metrics_reduced = reduce_metrics_over_batching_axes(
        active_train_metrics, train_strategy, "train"
    )
    # logger.debug(f"active_train_metrics_reduced shapes: {get_pytree_shapes(active_train_metrics_reduced)}")

    current_train_metrics_full = _construct_full_metrics_tree(
        active_metrics_tree=active_train_metrics_reduced,  # Pass the reduced metrics
        active_hyp_indices=active_hyp_indices,
        num_hyperparams=num_hyperparams,
        last_full_metrics_tree=None,  # Always fill train metrics from scratch (no carry-over of NaNs)
        fill_completed_with_nan=True,
    )

    train_agg = aggregate_metrics_per_hyperparam(
        metrics_dict=current_train_metrics_full,
        num_hyperparams=num_hyperparams,
        # keep_full={},  # Not used in this version
        skip_metric_key=("valid",),  # Matches original implementation
    )

    # --- Reduce active_eval_metrics over its batching dimensions ---
    # This will output metrics with shape (HP, OriginalMetricDims...) or (HP,)
    active_eval_metrics_reduced = reduce_metrics_over_batching_axes(
        active_eval_metrics, eval_strategy, "eval"
    )
    # logger.debug(f"active_eval_metrics_reduced shapes: {get_pytree_shapes(active_eval_metrics_reduced)}")

    current_eval_metrics_full = _construct_full_metrics_tree(
        active_metrics_tree=active_eval_metrics_reduced,  # Pass the reduced metrics
        active_hyp_indices=active_hyp_indices,
        num_hyperparams=num_hyperparams,
        last_full_metrics_tree=last_full_eval_metrics,
        fill_completed_with_nan=False,  # For eval, carry over previous values
    )

    eval_agg = aggregate_metrics_per_hyperparam(
        metrics_dict=current_eval_metrics_full,
        num_hyperparams=num_hyperparams,
        # keep_full={},  # Not used in this version
        skip_metric_key=("valid",),  # Matches original implementation
    )

    # logger.debug(f"eval_agg shapes: {jax.tree_util.tree_map(lambda x: x.shape if hasattr(x, 'shape') else type(x), eval_agg)}")
    # logger.debug(f"eval_agg shapes: {get_pytree_shapes(eval_agg)}")

    # Save aggregated metrics
    logger.debug(
        f"Saving aggregated metrics for eval_step {eval_step} to base_exp_path: {base_exp_path}"
    )
    save_metrics(base_exp_path, train_agg, eval_agg, eval_step)

    return current_eval_metrics_full, eval_agg


def _compute_auc_at_budget(
    timesteps: list[float],
    means: jnp.ndarray,
    budget_steps: float | None,
    target_return: float | None,
    *,
    baseline: str = "min",   # "min" | "first"
    eps: float = 1e-8,
) -> float:
    """
    Baseline-normalized AUC@budget in [0,1]:
      1) sort (t, mean)
      2) clip to budget with linear interpolation
      3) choose baseline B (min or first value in window)
      4) choose target T > B (use target_return if valid; otherwise fallback to high quantile/max)
      5) normalize: z = clip((y - B) / (T - B), 0, 1)
      6) AUC = trapezoid(z) / duration
    Robust to negative rewards and varying scales.
    """
    if not timesteps or means.size == 0:
        return 0.0

    # Sort
    ts_sorted = sorted(float(t) for t in timesteps)
    vals_sorted = [float(means[timesteps.index(float(t))]) for t in ts_sorted]
    ts = jnp.array(ts_sorted, dtype=jnp.float32)
    ys = jnp.array(vals_sorted, dtype=jnp.float32)

    # Determine budget window
    end_t = float(ts[-1]) if budget_steps is None else min(float(budget_steps), float(ts[-1]))
    start_t = float(ts[0])
    duration = end_t - start_t
    if duration <= eps:
        return 0.0

    # If budget is inside the series, interpolate an endpoint at end_t
    if end_t < float(ts[-1]):
        idx = int(jnp.searchsorted(ts, end_t, side="right") - 1)
        idx = max(0, min(idx, ts.size - 2))
        t0, t1 = float(ts[idx]), float(ts[idx + 1])
        y0, y1 = float(ys[idx]), float(ys[idx + 1])
        if abs(t1 - t0) < eps:
            y_end = y1
        else:
            y_end = y0 + (y1 - y0) * ((end_t - t0) / (t1 - t0))
        ts_w = jnp.concatenate([ts[: idx + 1], jnp.array([end_t], dtype=jnp.float32)])
        ys_w = jnp.concatenate([ys[: idx + 1], jnp.array([y_end], dtype=jnp.float32)])
    else:
        ts_w = ts
        ys_w = ys

    # 3) Baseline B
    if baseline == "first":
        B = float(ys_w[0])
    else:  # "min" over window (more robust if early values fluctuate)
        B = float(jnp.min(ys_w))

    # 4) Target T (must be > B)
    T = None
    if target_return is not None:
        T = float(target_return)

    # If provided target invalid, fallback to a high quantile / max above baseline
    if T is None or T <= B + eps:
        q90 = float(jnp.quantile(ys_w, 0.9))
        maxv = float(jnp.max(ys_w))
        # pick something above B; prefer q90 if it’s >B; else max; else B+1
        cand = q90 if q90 > B + eps else (maxv if maxv > B + eps else B + 1.0)
        T = cand

    denom = max(T - B, eps)

    # 5) Normalize by improvement over baseline, clipped to [0,1]
    z = jnp.clip((ys_w - B) / denom, 0.0, 1.0)

    # 6) Average normalized performance over time
    auc = jnp.sum(0.5 * (z[1:] + z[:-1]) * (ts_w[1:] - ts_w[:-1]))
    return float(auc / duration)


def _compute_time_to_target(
    timesteps: list[float],
    means: jnp.ndarray,                 # SAME ORDER as timesteps
    target_return: float | None,
    *,
    budget_steps: float | None = None,
) -> float:
    """RL-standard 'first crossing' with robust sorting and guards.
    Returns the first (interpolated) timestep where mean crosses target from below.
    Falls back to budget/last step. Never returns negative time.
    """
    # Basic guards
    if target_return is None or not timesteps or means.size == 0:
        return float("inf")

    # 1) Pair, sort by time, and drop NaNs
    #    (avoid float equality/index lookups that can misalign data)
    pairs = [(float(t), float(m)) for t, m in zip(timesteps, list(means))]
    pairs = sorted(pairs, key=lambda x: x[0])
    # Drop NaNs in means
    pairs = [(t, m) for (t, m) in pairs if not (np.isnan(m) or np.isnan(t))]
    if not pairs:
        return float("inf")

    ts = jnp.array([p[0] for p in pairs], dtype=jnp.float32)
    vals = jnp.array([p[1] for p in pairs], dtype=jnp.float32)

    # 2) Clip to budget (keep points <= budget, and optionally the first > budget for interpolation)
    if budget_steps is not None:
        B = float(budget_steps)
        # keep all <= B, and one extra point right after B (for interpolation)
        keep_idx = [i for i, t in enumerate(ts) if float(t) <= B]
        first_after = next((i for i, t in enumerate(ts) if float(t) > B), None)
        if first_after is not None:
            keep_idx.append(first_after)
        if keep_idx:
            keep_idx = sorted(set(keep_idx))
            ts = ts[keep_idx]
            vals = vals[keep_idx]
        # ensure budget is within [first,last]
        if ts.size == 0:
            return max(0.0, B)

    # 3) Early hit: already at/above target at the first timestamp
    target = float(target_return)
    if float(vals[0]) >= target:
        return max(0.0, float(ts[0]))

    # 4) Scan segments for a proper crossing: y0 < target <= y1
    for i in range(len(ts) - 1):
        t0, t1 = float(ts[i]), float(ts[i + 1])
        y0, y1 = float(vals[i]), float(vals[i + 1])

        # If budget is set and the segment starts beyond it, stop
        if budget_steps is not None and t0 >= float(budget_steps):
            break

        # Crossing condition from below
        if (y0 < target) and (y1 >= target):
            if abs(y1 - y0) < 1e-12 or abs(t1 - t0) < 1e-12:
                return max(0.0, t1)
            frac = (target - y0) / (y1 - y0)  # in (0,1]
            t_cross = t0 + frac * (t1 - t0)

            # Respect budget cap
            if budget_steps is not None:
                t_cross = min(t_cross, float(budget_steps))

            return max(0.0, float(t_cross))

    # 5) Not reached → return budget (if provided) or last step
    if budget_steps is not None:
        return max(0.0, min(float(ts[-1]), float(budget_steps)))
    return max(0.0, float(ts[-1]))


def _tail_smoothness(
    stats: dict,
    tail_frac: float = 0.10,
    min_points: int = 5,
) -> float:
    """
    Robust smoothness over the tail (last ~10%) of the run.
    tail_smoothness = clip(1 - IQR/|median|, 0, 1)

    Returns 0.0 if insufficient points or no valid means.
    """
    # Gather all (t, mean) pairs
    ts = sorted(
        [float(t) for t, v in stats.items() if isinstance(v, dict) and ("mean" in v)]
    )
    if not ts:
        return 0.0

    k = max(int(len(ts) * tail_frac), min_points, 1)
    tail_ts = ts[-k:]

    vals = []
    for t in tail_ts:
        v = stats.get(t, stats.get(str(t), {}))
        if isinstance(v, dict) and ("mean" in v):
            vals.append(float(v["mean"]))

    if len(vals) < 2:
        return 0.0

    arr = jnp.array(vals, dtype=jnp.float32)
    med = jnp.nanmedian(arr)
    q25 = jnp.nanpercentile(arr, 25)
    q75 = jnp.nanpercentile(arr, 75)
    iqr = q75 - q25
    smooth = 1.0 - (iqr / (jnp.abs(med) + 1e-8))
    return float(jnp.clip(smooth, 0.0, 1.0))


def _compute_learning_curve_metrics(return_time_series):
    """Compute metrics related to learning curve stability and performance over time."""
    metrics = {}

    # Filter out timesteps where the 'mean' is missing or NaN to avoid errors
    valid_points = {
        t: s
        for t, s in return_time_series.items()
        if isinstance(s, dict) and "mean" in s and not np.isnan(s.get("mean"))
    }

    if len(valid_points) < 3:
        logger.debug("Insufficient valid data points (< 3) for learning curve analysis.")
        return {"insufficient_data": True}  # Signal to caller

    # Extract sorted timesteps and mean returns
    timesteps = sorted(valid_points.keys())
    mean_returns = jnp.array([valid_points[t]["mean"] for t in timesteps])

    # Find peak return and its timestamp
    peak_idx = jnp.argmax(mean_returns)
    peak_return = mean_returns[peak_idx]
    peak_timestep = timesteps[peak_idx]

    # Calculate final return (average of last 10% of data points)
    final_window = max(1, int(len(mean_returns) * 0.1))
    final_return = jnp.mean(mean_returns[-final_window:])

    # Catastrophic forgetting metrics
    drop_from_peak = peak_return - final_return
    if jnp.abs(peak_return) > 1e-8:
        relative_drop = drop_from_peak / jnp.abs(peak_return) * 100
    else:
        relative_drop = jnp.float32(0.0)

    forgetting_score = jnp.clip(relative_drop / 100, 0, 1)
    stability_score = 1.0 - forgetting_score

    # Return timing (did performance peak early or late?)
    peak_time_ratio = peak_timestep / (timesteps[-1] + 1e-8)

    # Learning progress metrics (early learning rate)
    mid_point_idx = len(mean_returns) // 2
    if mid_point_idx < len(mean_returns) and len(mean_returns) > 0:
        first_half_improvement = mean_returns[mid_point_idx] - mean_returns[0]
        first_half_duration = timesteps[mid_point_idx] - timesteps[0]

        if first_half_duration > 0:
            early_learning_rate = first_half_improvement / first_half_duration
        else:
            early_learning_rate = jnp.float32(0.0)
    else:
        early_learning_rate = jnp.float32(0.0)

    # Package all metrics
    metrics["peak_return"] = float(peak_return)
    metrics["peak_timestep"] = int(peak_timestep)
    metrics["peak_time_ratio"] = float(peak_time_ratio)
    metrics["final_return"] = float(final_return)
    metrics["drop_from_peak"] = float(drop_from_peak)
    metrics["relative_drop_percent"] = float(relative_drop)
    metrics["forgetting_score"] = float(forgetting_score)
    metrics["stability_score"] = float(stability_score)
    metrics["early_learning_rate"] = float(early_learning_rate)

    # Classification of learning behavior
    if peak_time_ratio > 0.9:
        metrics["learning_pattern"] = "continuously_improving"
    elif peak_time_ratio < 0.3:
        if forgetting_score > 0.3:
            metrics["learning_pattern"] = "early_peak_significant_forgetting"
        else:
            metrics["learning_pattern"] = "early_peak_stable"
    else:
        if forgetting_score > 0.3:
            metrics["learning_pattern"] = "mid_training_regression"
        else:
            metrics["learning_pattern"] = "mid_peak_stable"

    return metrics


def summarize_hyperparam_performance(
    return_trackers: list[HyperparamReturns],
    top_n: int = 5,
    sort_by: str = "peak",
    *,
    budget_steps: float | None = None,          # AUC budget in env steps
    target_return: float | None = None,         # For AUC normalization and t@target
    final_window_frac: float = 0.1,
    final_window_min_points: int = 3,
    save_csv_path: Path | None = None,
) -> list[dict[str, Any]]:
    """
    Summarize performance of hyperparameter configurations for RL.
    Lean metrics; stability_score removed; tail_smoothness added.
    """
    groups = []
    for tracker_idx, tr in enumerate(return_trackers):
        groups.append({
            "tracker_original_idx_in_list": tracker_idx,
            "sample_id": tr.sample_id,
            "hyperparam": tr.hyperparams,
            "return": tr.episode_returns,
            "return_stats": tr.return_stats,
        })

    logger.info("\n===== HYPERPARAMETER PERFORMANCE SUMMARY =====")
    logger.info(f"Total configs used during metric summary gathering: {len(groups)}")

    summaries: list[dict[str, Any]] = []
    for g in groups:
        hyp = g["hyperparam"]
        stats = g["return_stats"]
        sample_id = g["sample_id"]
        idx_in_list = g["tracker_original_idx_in_list"]

        if not stats:
            logger.warning(f"No return_stats for sample_id {sample_id}. Skipping.")
            continue

        timestamps = sorted(stats.keys())
        if not timestamps:
            logger.warning(f"No timesteps in return_stats for sample_id {sample_id}. Skipping.")
            continue

        lc = _compute_learning_curve_metrics(stats)

        # Defaults
        final_perf = 0.0
        final_median = 0.0
        peak_perf = 0.0
        peak_ci_lo = 0.0
        peak_ci_hi = 0.0
        forgetting_score = 1.0
        tail_smooth = 0.0
        auc_at_budget = 0.0
        t_at_target = float("inf")
        peak_time_ratio = 0.0

        if lc.get("insufficient_data", False):
            # Fallback: use last and max valid means
            valid_means = [
                s.get("mean")
                for s in stats.values()
                if isinstance(s, dict) and "mean" in s and not np.isnan(s["mean"])
            ]
            valid_meds = [
                s.get("median")
                for s in stats.values()
                if isinstance(s, dict) and "median" in s and not np.isnan(s["median"])
            ]
            if valid_means:
                final_perf = float(valid_means[-1])
                final_median = float(valid_meds[-1]) if valid_meds else final_perf
                peak_perf = float(max(valid_means))
            else:
                final_perf = peak_perf = -float("inf")
            # tail_smooth stays 0.0; AUC/t@target omitted
        else:
            # Final metrics over last window
            n_pts = len(timestamps)
            tail_n = max(int(n_pts * final_window_frac), final_window_min_points, 1)
            tail_ts = timestamps[-tail_n:]

            tail_means = [
                stats[t]["mean"]
                for t in tail_ts
                if isinstance(stats.get(t), dict) and "mean" in stats[t]
            ]
            tail_meds = [
                stats[t]["median"]
                for t in tail_ts
                if isinstance(stats.get(t), dict) and "median" in stats[t]
            ]

            final_perf = float(jnp.mean(jnp.array(tail_means))) if tail_means else 0.0
            final_median = float(jnp.mean(jnp.array(tail_meds))) if tail_meds else 0.0

            # Forgetting and peak info from lc
            forgetting_score = float(lc.get("forgetting_score", 1.0))
            peak_perf = float(lc.get("peak_return", 0.0))
            peak_time_ratio = float(lc.get("peak_time_ratio", 0.0))
            peak_ts = lc.get("peak_timestep")
            peak_stats = stats.get(peak_ts, {}) if peak_ts is not None else {}
            if isinstance(peak_stats, dict):
                peak_ci_lo = float(peak_stats.get("bootstrap_ci_lower", peak_perf))
                peak_ci_hi = float(peak_stats.get("bootstrap_ci_upper", peak_perf))

            # Tail smoothness (robust)
            tail_smooth = _tail_smoothness(stats, tail_frac=final_window_frac, min_points=final_window_min_points)

            # Build sequences for AUC@budget and t@target
            all_ts, all_means = [], []
            for t in timestamps:
                s = stats.get(t)
                if isinstance(s, dict) and "mean" in s:
                    all_ts.append(float(t))
                    all_means.append(float(s["mean"]))

            if len(all_ts) > 0:
                means_arr = jnp.array(all_means, dtype=jnp.float32)

                # ---- choose effective target if none provided ----
                if target_return is None:
                    _eff_target = 0.9 * float(peak_perf) if float(peak_perf) > 0 else float(final_perf)
                else:
                    _eff_target = float(target_return)

                # Normalized AUC@budget ∈ [0,1]
                auc_at_budget = _compute_auc_at_budget(all_ts, means_arr, budget_steps, _eff_target)

                # Finite, safe t@Target using same target policy
                t_at_target = _compute_time_to_target(
                    all_ts,
                    means_arr,
                    _eff_target,
                    budget_steps=budget_steps,
                )
            else:
                auc_at_budget = 0.0
                t_at_target = float("inf")

        summaries.append({
            "tracker_original_idx_in_list": idx_in_list,
            "sample_id": sample_id,
            "final_performance": final_perf,
            "final_performance_median": final_median,
            "peak_performance": peak_perf,
            "peak_ci_lower": peak_ci_lo,
            "peak_ci_upper": peak_ci_hi,
            "tail_smoothness": tail_smooth,         # NEW
            "forgetting_score": forgetting_score,   # Keep ONE of forgetting/stability
            "auc_at_budget": auc_at_budget,
            "t_at_target": t_at_target,
            "peak_time_ratio": peak_time_ratio,
            "hyperparam": hyp,
            "learning_curve_metrics": lc,
            "return_stats": stats,
        })

    if not summaries:
        logger.error("No valid hyperparameter configurations with sufficient statistics found.")
        return []

    # Sorting (map legacy 'stability' to tail_smoothness)
    sort_key_map = {
        "final": lambda x: x["final_performance"],
        "final_median": lambda x: x["final_performance_median"],
        "auc": lambda x: x["auc_at_budget"],
        "stability": lambda x: x["tail_smoothness"],     # backward-compat
        "tail_smoothness": lambda x: x["tail_smoothness"],
        "consistency": lambda x: x.get("ci_tail_width", float("inf")),  # ascending
        "peak": lambda x: x["peak_performance"],
        "combined": lambda x: x["peak_performance"],     # backward-compat
        "t_at_target": lambda x: x["t_at_target"],       # ascending (smaller is better)
        "forgetting": lambda x: x["forgetting_score"],   # ascending if you choose to minimize externally
    }
    sort_key_lambda = sort_key_map.get(sort_by, sort_key_map["peak"])
    should_reverse_sort = sort_by not in ("consistency", "t_at_target")
    sorted_summaries = sorted(summaries, key=sort_key_lambda, reverse=should_reverse_sort)

    # Optional CSV export
    if save_csv_path:
        rows_for_df = []
        for s in sorted_summaries:
            row = {k: v for k, v in s.items() if not isinstance(v, (dict, list))}
            row.update(s.get("hyperparam", {}))
            rows_for_df.append(row)
        df = pd.DataFrame(rows_for_df)
        perf_cols = [
            "sample_id",
            "peak_performance",
            "final_performance",
            "tail_smoothness",
            "auc_at_budget",
            "forgetting_score",
            "t_at_target",
            "peak_time_ratio",
        ]
        hyperparam_cols = sorted([c for c in df.columns if c not in perf_cols])
        out_cols = [c for c in perf_cols if c in df.columns] + hyperparam_cols
        df[out_cols].to_csv(save_csv_path, index=False, float_format="%.4g")

    # Console table (lean)
    headers = [
        "UID",
        "Peak.Mean",
        "Peak.CI",
        "End.Mean",
        "Tail.Smooth",
        "Forget",
        "AUC@Budget",
        "t@Target",
        "Hyperparameters",
    ]
    table_data = []
    for s in sorted_summaries[: min(top_n, len(sorted_summaries))]:
        hp = s["hyperparam"]
        peak_ci_str = f"[{s['peak_ci_lower']:.2f},{s['peak_ci_upper']:.2f}]"
        hp_parts = []
        for k, v in hp.items():
            if isinstance(v, float):
                hp_parts.append(f"{k}={v:.1e}")
            elif isinstance(v, (bool, int, np.integer)):
                hp_parts.append(f"{k}={v}")
            elif isinstance(v, (list, tuple)):
                hp_parts.append(f"{k}={str(v)}")
            else:
                hp_parts.append(f"{k}={str(v)}")
        row = [
            s["sample_id"],
            f"{s['peak_performance']:.2f}",
            peak_ci_str,
            f"{s['final_performance']:.2f}",
            f"{s['tail_smoothness']:.2f}",
            f"{s['forgetting_score']:.2f}",
            f"{s['auc_at_budget']:.2f}",
            ("∞" if np.isinf(s["t_at_target"]) else f"{s['t_at_target']:.0f}"),
            ", ".join(hp_parts),
        ]
        table_data.append(row)

    logger.info("\n" + tabulate(table_data, headers=headers, tablefmt="simple", floatfmt=".2f"))
    return sorted_summaries


def get_final_and_peak_perf_from_hyperparam_trackers(
    return_trackers: list[HyperparamReturns],
    initial_num_hyperparams: int,  # Number of HPs in the original batch passed to run_unified_...
) -> tuple[list[float], list[float]]:
    # This will generate a list of summary dictionaries.
    # If run_unified_algorithm_experiment processed N hyperparams,
    # return_trackers will have N items, and all_hp_summaries will have N items.
    all_hp_summaries = summarize_hyperparam_performance(
        return_trackers, top_n=max(1, initial_num_hyperparams)
    )
    logger.info(
        f"Summarized HP performance. Received {len(all_hp_summaries)} summaries for {initial_num_hyperparams} expected HPs."
    )

    # Initialize lists to store metrics in the original order of hyperparams
    final_mean_returns_per_hp = [0.0] * initial_num_hyperparams
    peak_performances_per_hp = [0.0] * initial_num_hyperparams

    if not all_hp_summaries:
        logger.warning("No summaries generated. Returning default 0.0s for all HPs.")
        return final_mean_returns_per_hp, peak_performances_per_hp

    # Map summaries to their original positions using 'tracker_original_idx_in_list'
    # which corresponds to the index in the `return_trackers` list.
    for summary_item in all_hp_summaries:
        # 'tracker_original_idx_in_list' is the 0-based index from the input `return_trackers` list
        original_batch_idx = summary_item.get("tracker_original_idx_in_list")
        sample_id_of_summary = summary_item.get("sample_id")  # For logging

        if original_batch_idx is None or not (0 <= original_batch_idx < initial_num_hyperparams):
            logger.warning(
                f"Summary for sample_id {sample_id_of_summary} has invalid/missing "
                f"'tracker_original_idx_in_list': {original_batch_idx}. Expected range [0, {initial_num_hyperparams - 1}]. Skipping."
            )
            continue

        final_perf = float(summary_item.get("final_performance", 0.0))
        peak_perf = float(summary_item.get("peak_performance", 0.0))

        final_mean_returns_per_hp[original_batch_idx] = final_perf
        peak_performances_per_hp[original_batch_idx] = peak_perf
        logger.info(
            f"For HP at original_batch_idx={original_batch_idx} (Sample ID: {sample_id_of_summary}): "
            f"Final={final_perf:.2f}, Peak={peak_perf:.2f}"
        )

    logger.info(f"Final list 'final_mean_returns_per_hp': {final_mean_returns_per_hp}")
    logger.info(f"Final list 'peak_performances_per_hp': {peak_performances_per_hp}")

    return final_mean_returns_per_hp, peak_performances_per_hp


def calculate_learning_trend_indicators(
    hp_runtime_states: list[HPRuntimeState],
    return_trackers: list[Any],  # List[HyperparamReturns]
    # This dictionary is mutable and will be updated by this function for active HPs.
    last_known_indicators_state: dict[int, str],
    trend_threshold: float = 0.01,
) -> list[str]:  # Returns the list of indicators for *current display*.
    """
    Calculates a learning trend indicator for each HP.
    - For active HPs, calculates the current trend and updates `last_known_indicators_state`.
    - For inactive HPs, uses the value from `last_known_indicators_state`.
    Returns a list of formatted strings suitable for immediate display.
    """
    current_display_indicators = [" "] * len(hp_runtime_states)  # Pre-allocate for all HPs
    # Efficiently map sample_id (original_index) to its tracker
    tracker_map: dict[int, Any] = {tracker.sample_id: tracker for tracker in return_trackers}

    for i, hp_state in enumerate(hp_runtime_states):
        hp_original_idx = hp_state.original_index
        # Default indicator: dim dot, used if no data or inactive with no history
        default_display_indicator = f"{Style.DIM}·{Style.RESET_ALL}"

        if not hp_state.is_active:
            # For inactive HPs, use last known indicator. If never recorded, use default.
            current_display_indicators[i] = last_known_indicators_state.get(
                hp_original_idx, default_display_indicator
            )
            continue

        # HP is active: calculate its current trend.
        # Start with default; will be updated if trend can be calculated.
        calculated_indicator_str_for_active_hp = default_display_indicator
        indicator_symbol = "·"  # Default symbol
        color_code = Style.DIM  # Default color

        tracker = tracker_map.get(hp_original_idx)
        # Check if tracker exists and has return_stats
        if tracker and tracker.return_stats:
            try:
                # Timesteps are keys in return_stats. Convert to float for reliable sorting.
                sorted_timesteps = sorted([float(k) for k in tracker.return_stats.keys()])
            except ValueError:
                # Fallback if keys aren't purely numeric (should not happen with current setup)
                sorted_timesteps = sorted(tracker.return_stats.keys())

            if len(sorted_timesteps) >= 2:  # Need at least two points to determine a trend
                try:
                    # Get the last two timesteps
                    last_ts_key = sorted_timesteps[-1]
                    second_last_ts_key = sorted_timesteps[-2]

                    # Retrieve stats for these timesteps. Handle potential float/str key issues.
                    current_stats_dict = tracker.return_stats.get(
                        last_ts_key, tracker.return_stats.get(str(last_ts_key))
                    )
                    previous_stats_dict = tracker.return_stats.get(
                        second_last_ts_key,
                        tracker.return_stats.get(str(second_last_ts_key)),
                    )

                    current_mean_val = (
                        current_stats_dict.get("mean") if current_stats_dict else None
                    )
                    previous_mean_val = (
                        previous_stats_dict.get("mean") if previous_stats_dict else None
                    )

                    if current_mean_val is not None and previous_mean_val is not None:
                        current_mean = float(current_mean_val)
                        previous_mean = float(previous_mean_val)
                        epsilon = 1e-9  # For safe comparison against zero

                        # Determine trend based on relative change
                        if abs(previous_mean) < epsilon:  # Previous mean is effectively zero
                            if (
                                current_mean
                                > previous_mean + (abs(previous_mean) * trend_threshold) + epsilon
                            ):
                                indicator_symbol, color_code = "↑", Fore.GREEN
                            elif (
                                current_mean
                                < previous_mean - (abs(previous_mean) * trend_threshold) - epsilon
                            ):
                                indicator_symbol, color_code = "↓", Fore.RED
                            else:  # No significant change from (near) zero
                                indicator_symbol, color_code = "→", Fore.YELLOW
                        else:  # Normal case: previous_mean is not zero
                            relative_change = (current_mean - previous_mean) / abs(previous_mean)
                            if relative_change > trend_threshold:
                                indicator_symbol, color_code = "↑", Fore.GREEN
                            elif relative_change < -trend_threshold:
                                indicator_symbol, color_code = "↓", Fore.RED
                            else:  # Change is not significant
                                indicator_symbol, color_code = "→", Fore.YELLOW

                        calculated_indicator_str_for_active_hp = (
                            f"{color_code}{indicator_symbol}{Style.RESET_ALL}"
                        )
                except Exception:
                    # If any error occurs during trend calculation defaults to the dim dot.
                    # `calculated_indicator_str_for_active_hp` remains `default_display_indicator`.
                    pass

        # Store the calculated (or default if calculation failed) indicator for this *active* HP
        last_known_indicators_state[hp_original_idx] = calculated_indicator_str_for_active_hp
        # Set the indicator for current display
        current_display_indicators[i] = calculated_indicator_str_for_active_hp

    return current_display_indicators


from typing import Any  # Ensure these are present


def get_metrics_for_optuna_from_hyperparam_trackers(
    return_trackers: list[HyperparamReturns],
    initial_num_hyperparams: int,
    objective_names: list[str],
) -> list[dict[str, float]]:
    # Compute summaries for all HPs
    all_hp_summaries = summarize_hyperparam_performance(
        return_trackers, top_n=max(1, initial_num_hyperparams)
    )
    logger.info(
        f"Summarized HP performance. Received {len(all_hp_summaries)} summaries for {initial_num_hyperparams} expected HPs."
    )

    # Prepare results in original HP order
    results_in_order: list[dict[str, float] | None] = [None] * initial_num_hyperparams

    if not all_hp_summaries:
        logger.warning("No summaries generated. Returning default metric dicts.")
        def_val = {}
        for name in objective_names:
            if name == "forgetting_score":
                def_val[name] = 1.0
            else:
                # tail_smoothness and others are 'maximize' metrics → default worst is 0.0
                def_val[name] = 0.0
        return [def_val.copy() for _ in range(initial_num_hyperparams)]

    for s in all_hp_summaries:
        orig_idx = s.get("tracker_original_idx_in_list")
        sid = s.get("sample_id")
        if orig_idx is None or not (0 <= orig_idx < initial_num_hyperparams):
            logger.warning(
                f"Summary for sample_id {sid} has invalid/missing original index: {orig_idx}. Skipping."
            )
            continue

        metric_dict: dict[str, float] = {}
        for name in objective_names:
            # Backward-compat mapping
            key = name
            if name == "stability_score":
                key = "tail_smoothness"  # map stability -> tail_smoothness (NOTE removed stability_score and kept forgetting score only)

            default_val = 1.0 if key == "forgetting_score" else 0.0
            val = s.get(key, default_val)

            try:
                metric_dict[name] = float(val)
            except (TypeError, ValueError):
                logger.warning(
                    f"Could not convert metric '{name}' (mapped to '{key}') value '{val}' to float; using default {default_val}."
                )
                metric_dict[name] = float(default_val)

        if results_in_order[orig_idx] is not None:
            logger.warning(f"Duplicate summary for index {orig_idx}; overwriting.")
        results_in_order[orig_idx] = metric_dict
        logger.info(f"HP idx={orig_idx} (Sample ID: {sid}) metrics: {metric_dict}")

    # Fill any missing slots
    filled: list[dict[str, float]] = []
    for i in range(initial_num_hyperparams):
        if results_in_order[i] is not None:
            filled.append(results_in_order[i])
        else:
            def_val = {}
            for name in objective_names:
                key = "tail_smoothness" if name == "stability_score" else name
                def_val[name] = 1.0 if key == "forgetting_score" else 0.0
            filled.append(def_val)
            logger.warning(f"No summary for HP index {i}; returning defaults {def_val}")
    return filled
