import logging
from typing import Any

import jax
import jax.numpy as jnp

from hyperlax.base_types import GenericState, HPRuntimeState
from hyperlax.layout.axes import DistributionStrategy
from hyperlax.logger.return_tracker import HyperparamReturns

logger = logging.getLogger(__name__)


def build_active_learner_state(
    full_learner_state: GenericState,
    active_indices: jnp.ndarray,
    num_total_hyperparams: int,
    train_strategy: DistributionStrategy,
) -> GenericState:
    """
    Extracts active parts of the learner state for continuing training.
    Preserves the original axis layout - does NOT move HP to axis 0 to stay axes aware
    """
    hp_axis_in_full_state = train_strategy.get_axis_position("hyperparam")

    def slice_leaf(leaf: Any) -> Any:
        if not hasattr(leaf, "shape") or not leaf.shape:
            return leaf
        # Check if this leaf is batched by HP at the expected axis
        if (
            leaf.ndim > hp_axis_in_full_state
            and leaf.shape[hp_axis_in_full_state] == num_total_hyperparams
        ):
            # This is a per-HP leaf - slice along the HP axis using active_indices
            slicing_op = [slice(None)] * leaf.ndim
            slicing_op[hp_axis_in_full_state] = active_indices
            sliced_leaf = leaf[tuple(slicing_op)]
            return sliced_leaf
        else:
            return leaf

    return jax.tree_util.tree_map(slice_leaf, full_learner_state)


def sum_total_env_steps_per_hyperparam(
    total_env_steps_counter: jnp.ndarray,  # The full, distributed total_env_steps_counter
    strategy: DistributionStrategy,
) -> jnp.ndarray:
    """
    Sums =total_env_steps_counter= over the 'device' and 'update_batch' axes for each
    hyperparameter, assuming 'seed' and 'hyperparam' axes contain broadcasted/identical
    values. This yields a single total environment step count per hyperparameter configuration.

    Args:
        total_env_steps_counter: A JAX array representing the accumulated environment steps,
                                 with dimensions corresponding to the =strategy=.
                                 E.g., (Seed, HP, Device, UpdateBatch).
        strategy: The DistributionStrategy used for training, defining the axis layout.
        logger: Logger instance.

    Returns:
        A JAX array of shape (NumHPs,), where each element is the total environment steps
        for that specific hyperparameter configuration.
    """
    if not isinstance(total_env_steps_counter, jnp.ndarray):
        raise TypeError(
            f"Expected jnp.ndarray for total_env_steps_counter, got {type(total_env_steps_counter)}"
        )

    # logger.debug(f"DEBUG_SUM_STEPS: Processing total_env_steps_counter, initial shape: {total_env_steps_counter.shape}")

    # Pre-calculate positions of all relevant axes defined in the strategy
    axis_name_to_pos = {axis_spec.name: axis_spec.in_axes for axis_spec in strategy.axes}

    hp_axis_pos = axis_name_to_pos.get("hyperparam")
    seed_axis_pos = axis_name_to_pos.get("seed")
    device_axis_pos = axis_name_to_pos.get("device")
    update_batch_axis_pos = axis_name_to_pos.get("update_batch")

    if hp_axis_pos is None:
        raise ValueError("Hyperparam axis not found in strategy.")
    if seed_axis_pos is None:
        raise ValueError("Seed axis not found in strategy.")
    # device and update_batch might not exist in all strategies (e.g. simple tests)
    # if device_axis_pos is None: raise ValueError("Device axis not found in strategy.")

    # --- Step 1: Select from Seed and UpdateBatch axes (take the 0-th element) ---
    slicing_indices = [slice(None)] * total_env_steps_counter.ndim

    # For Seed and UpdateBatch axes, select the 0th element.
    if seed_axis_pos is not None and seed_axis_pos < total_env_steps_counter.ndim:
        slicing_indices[seed_axis_pos] = 0
    if update_batch_axis_pos is not None and update_batch_axis_pos < total_env_steps_counter.ndim:
        slicing_indices[update_batch_axis_pos] = 0

    # Apply the slice. This reduces the dimensions of Seed and UpdateBatch to 1 (or removes them if they were size 1).
    selected_from_broadcasted_axes = total_env_steps_counter[tuple(slicing_indices)]
    # logger.debug(f"DEBUG_SUM_STEPS: After selecting 0th element from Seed/UB: {selected_from_broadcasted_axes.shape}")

    # --- Step 2: Sum over the Device axis ---
    summed_over_devices = selected_from_broadcasted_axes
    if device_axis_pos is not None:
        actual_device_axis_pos = device_axis_pos
        if seed_axis_pos is not None and seed_axis_pos < device_axis_pos:
            actual_device_axis_pos -= (
                1  # Device axis shifts left by 1 if seed was before it and sliced
            )
        if update_batch_axis_pos is not None and update_batch_axis_pos < device_axis_pos:
            actual_device_axis_pos -= 1

        # Sum over the device axis if it exists in the remaining dimensions
        if actual_device_axis_pos < summed_over_devices.ndim:
            summed_over_devices = jnp.sum(summed_over_devices, axis=actual_device_axis_pos)
            # logger.debug(f"DEBUG_SUM_STEPS: After summing over device: {summed_over_devices.shape}")

    # --- Step 3: Ensure HP dimension is at axis 0 and only the HP dimension remains ---
    actual_hp_axis_pos = hp_axis_pos
    if seed_axis_pos is not None and seed_axis_pos < hp_axis_pos:
        actual_hp_axis_pos -= 1
    if update_batch_axis_pos is not None and update_batch_axis_pos < hp_axis_pos:
        actual_hp_axis_pos -= 1
    if (
        device_axis_pos is not None and device_axis_pos < hp_axis_pos
    ):  # Account for device dim shift
        actual_hp_axis_pos -= 1

    final_result = summed_over_devices
    if final_result.ndim > 1:  # If there are dimensions other than HP remaining
        # Move HP axis to position 0
        if actual_hp_axis_pos != 0:
            final_result = jnp.moveaxis(final_result, actual_hp_axis_pos, 0)
        # Flatten all dimensions after the HP dimension
        final_result = final_result.reshape(final_result.shape[0], -1).squeeze(
            axis=-1
        )  # Squeeze if it results in (HP,1)
        # logger.debug(f"DEBUG_SUM_STEPS: After flattening and squeezing: {final_result.shape}")
    elif final_result.ndim == 1:
        # Already (HP,)
        if (
            actual_hp_axis_pos != 0
        ):  # This case is unlikely if only 1 dim left, it must be the only one
            final_result = jnp.moveaxis(final_result, actual_hp_axis_pos, 0)
    else:  # Scalar case
        final_result = jnp.array([final_result])  # Make it (1,) for consistency

    # logger.debug(f"SUM_TOTAL_ENV_STEPS (LOG 4.2): Output summed_value (per HP): {final_result.tolist()}")
    return final_result


def get_env_step_counter(
    master_full_learner_state: Any,  # The complete learner state for ALL HPs
    original_hp_index: int,  # The ORIGINAL index of the HP being queried
    train_strategy: DistributionStrategy,
) -> int:
    """
    Extracts the total accumulated environment steps for a specific hyperparameter
    (identified by its original index) from the master full learner state.
    It sums =total_env_steps_counter= across all batching dimensions (Seed, Device, UpdateBatch, etc.)
    to get the total steps for the requested HP, leveraging the provided training strategy.
    """
    if not hasattr(master_full_learner_state, "total_env_steps_counter"):
        raise AttributeError("Learner state must have a 'total_env_steps_counter' attribute.")

    num_total_hp = master_full_learner_state.total_env_steps_counter.shape[
        train_strategy.get_axis_position("hyperparam")
    ]
    # if original_hp_index < num_total_hp:
    #     raise ValueError(f"To get env step counters we need: original_hp_index ({original_hp_index}) >= num_total_hp ({num_total_hp})")
    if original_hp_index >= num_total_hp:
        raise ValueError(
            f"To get env step counters we need: original_hp_index ({original_hp_index}) < num_total_hp ({num_total_hp})"
        )

    # Use the general sum_total_env_steps_per_hyperparam function from metrics.py.
    total_steps_for_all_hps = sum_total_env_steps_per_hyperparam(
        total_env_steps_counter=master_full_learner_state.total_env_steps_counter,
        strategy=train_strategy,
    )

    # Extract the scalar count for the requested HP using its original index.
    scalar_total_steps = total_steps_for_all_hps[original_hp_index]

    return int(scalar_total_steps)


def merge_active_state_into_full(
    master_full_learner_state: Any,
    active_learner_state_slice: Any,
    active_hp_original_indices: list[int],
    train_strategy: DistributionStrategy,
) -> Any:
    """
    Merges active learner state back into master state, respecting axis positions.
    Does NOT reorder the active slice - works with the axis layout as-is.
    """
    active_pos_to_orig_idx_map = {
        pos_in_active_slice: original_idx
        for pos_in_active_slice, original_idx in enumerate(active_hp_original_indices)
    }

    hp_axis_in_master = train_strategy.get_axis_position("hyperparam")

    def _update_leaf(full_leaf: Any, active_slice_leaf: Any) -> Any:
        # Sanity checks
        if not (
            hasattr(full_leaf, "shape")
            and hasattr(active_slice_leaf, "shape")
            and full_leaf.shape is not None
            and active_slice_leaf.shape is not None
        ):
            return full_leaf

        if not (full_leaf.ndim > hp_axis_in_master and active_slice_leaf.ndim > 0):
            return full_leaf

        # Get the HP dimension size to verify this is a per-HP leaf
        num_total_hps_in_master = master_full_learner_state.total_env_steps_counter.shape[
            hp_axis_in_master
        ]

        if not (
            full_leaf.shape[hp_axis_in_master] == num_total_hps_in_master
            and active_slice_leaf.shape[hp_axis_in_master] == len(active_hp_original_indices)
        ):
            # This leaf doesn't conform to per-HP batching, return unchanged
            return full_leaf

        updated_leaf = full_leaf
        for pos_in_active, original_idx in active_pos_to_orig_idx_map.items():
            # Extract data from the active slice using the HP axis position in the active slice
            # Build slice for the active slice where HP axis is at hp_axis_in_master
            active_slice_tuple = [slice(None)] * active_slice_leaf.ndim
            active_slice_tuple[hp_axis_in_master] = (
                pos_in_active  # HP is at same position in active slice
            )
            data_from_active_hp = active_slice_leaf[tuple(active_slice_tuple)]

            # Build slice for the master leaf where HP is at hp_axis_in_master
            master_idx_tuple = [slice(None)] * full_leaf.ndim
            master_idx_tuple[hp_axis_in_master] = original_idx

            updated_leaf = updated_leaf.at[tuple(master_idx_tuple)].set(data_from_active_hp)

        return updated_leaf

    return jax.tree_util.tree_map(
        _update_leaf, master_full_learner_state, active_learner_state_slice
    )


def extract_current_avg_returns_to_display(
    return_trackers: list[HyperparamReturns],
    hp_runtime_states: list[HPRuntimeState],
) -> list[float | None]:
    """Calculate current mean returns for each HP for progress display."""
    current_returns = [None] * len(hp_runtime_states)

    for i, hp_state in enumerate(hp_runtime_states):
        # hp_state.original_index is the correct index into the current return_trackers list
        tracker_idx = hp_state.original_index

        # Ensure the index is valid for the current list of trackers
        if 0 <= tracker_idx < len(return_trackers):
            tracker = return_trackers[tracker_idx]
            if tracker and tracker.episode_returns:
                # Get the latest timestep's stats
                latest_timestep = max(tracker.episode_returns.keys())
                if latest_timestep in tracker.return_stats:
                    current_returns[i] = tracker.return_stats[latest_timestep].get("mean")
        else:
            # This case should ideally not happen, but it's good to log
            logger.warning(
                f"Could not find a return tracker for hp_state with original_index {tracker_idx}. "
                f"len(return_trackers)={len(return_trackers)}."
            )

    return current_returns


def _calculate_milestones(target_total_steps: int, num_evaluation_milestones: int) -> list[int]:
    milestones = []
    if num_evaluation_milestones > 0:
        interval = target_total_steps // num_evaluation_milestones
        if interval == 0:
            interval = 1  # Ensure progress even for small total_steps
        # Loop to num_milestones - 1 to generate intermediate milestones
        for i in range(1, num_evaluation_milestones):
            milestones.append(min(i * interval, target_total_steps))

    # Always add the final target step, unless it's already the last one
    if not milestones or milestones[-1] < target_total_steps:
        milestones.append(target_total_steps)

    if not milestones and target_total_steps == 0:
        return [0]

    return sorted(list(set(milestones)))
