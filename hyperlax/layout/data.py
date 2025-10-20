from collections.abc import Callable
from typing import TYPE_CHECKING, Any, Union

import chex
import jax
import jax.numpy as jnp

if TYPE_CHECKING:
    from hyperlax.algo.dqn.struct_dqn import DQNVectorizedHyperparams
    from hyperlax.algo.ppo.struct_ppo import PPOVectorizedHyperparams

from hyperlax.layout.axes import DistributionStrategy


def _distribute_single_item(
    item_leaf: Any,
    strategy: DistributionStrategy,
) -> Any:
    """
    Distributes a single leaf item (e.g., initial parameter, scalar) across the strategy axes.
    The final shape of the item will match the ordering implied by the strategy's in_axes.
    It takes a scalar or (features,) leaf and expands it.
    """
    current_item = item_leaf

    # Determine the full target shape based on strategy and original item's features.
    # The dimensions specified by strategy.axes.in_axes will be populated.
    # Any other dimensions (e.g., features from item_leaf) will be appended.

    # Calculate the maximum index any 'in_axes' uses, to determine how many leading 1s we need.
    max_axis_index = max((axis.in_axes for axis in strategy.axes), default=-1)

    # Initialize a list for the target batching dimensions with 1s.
    # This ensures we have placeholders for all batching dimensions up to the max_axis_index.
    target_batch_dims = [1] * (max_axis_index + 1)

    # Fill in the actual sizes for the batching dimensions as per the strategy.
    for axis in strategy.axes:
        target_batch_dims[axis.in_axes] = axis.size

    # Construct the final target shape: (batching_dims_in_order) + (original_item_features)
    final_target_shape = tuple(target_batch_dims) + item_leaf.shape

    # # Expand the current_item to have enough leading '1' dimensions.
    # # This ensures `jnp.broadcast_to` can correctly expand all batch dimensions.
    # num_leading_ones_needed = (
    #     len(target_batch_dims) - len(item_leaf.shape)
    #     if item_leaf.ndim == 0
    #     else len(target_batch_dims) - item_leaf.ndim + item_leaf.shape.count(1)
    # )  # Correct for existing leading 1s
    if item_leaf.ndim < len(target_batch_dims):
        # Only add dimensions if the item doesn't already have them.
        # This handles cases where `item_leaf` might already be shaped (1,1,F) for example.
        for _ in range(len(target_batch_dims) - item_leaf.ndim):
            current_item = jnp.expand_dims(current_item, axis=0)

    # Perform the broadcast to the final target shape.
    broadcasted_item = jnp.broadcast_to(current_item, final_target_shape)

    return broadcasted_item


# `distribute_initial_components` remains the same as it correctly applies `_distribute_single_item`
# to each leaf of the input pytree.
# The `devices` argument passed to `distribute_initial_components` is no longer needed by `_distribute_single_item`
# itself, as `flax.jax_utils.replicate` is not used there directly.
# JAX's `pmap` transformation handles the physical distribution to device memory based on `in_axes`.
def distribute_initial_components(
    components: Any,
    strategy: DistributionStrategy,
    # devices: List[Any] # No longer needed here
) -> Any:
    return jax.tree_util.tree_map(
        lambda x: _distribute_single_item(x, strategy),  # Removed devices arg
        components,
    )


def distribute_hyperparam_struct_across_axes(
    hyperparams_struct: Union["PPOVectorizedHyperparams", "DQNVectorizedHyperparams"],
    strategy: DistributionStrategy,
    hyperparam_axis_name_in_strategy: str = "hyperparam",
) -> Union["PPOVectorizedHyperparams", "DQNVectorizedHyperparams"]:
    """
    Distributes a hyperparameter structure (already batched by its own HP dim)
    across the other axes defined in the strategy.
    The HP dimension in the input struct is assumed to be the *first* dimension (axis 0).
    """
    # 1. Determine the target position of the hyperparam axis in the final desired layout.
    hp_axis_pos = strategy.get_axis_position(hyperparam_axis_name_in_strategy)

    def _distribute_hp_leaf(hp_leaf_with_hp_dim: jnp.ndarray) -> jnp.ndarray:
        # hp_leaf_with_hp_dim has shape (NumHPs, original_features...).
        # e.g., (16, 5) if 16 HPs, and each HP has 5 features.
        num_hps = hp_leaf_with_hp_dim.shape[0]
        original_feature_dims = hp_leaf_with_hp_dim.shape[1:]

        # Build the full target shape.
        # This will be (batching_dims_in_order) + (original_feature_dims).
        max_axis_index = max((axis.in_axes for axis in strategy.axes), default=-1)
        target_batch_dims = [1] * (max_axis_index + 1)  # Placeholder for all batching dims

        for axis in strategy.axes:
            if axis.name == hyperparam_axis_name_in_strategy:
                target_batch_dims[axis.in_axes] = num_hps  # Use actual NumHPs from input
            else:
                target_batch_dims[axis.in_axes] = axis.size  # Use size from strategy

        final_target_shape = tuple(target_batch_dims) + original_feature_dims

        # The input `hp_leaf_with_hp_dim` has the HP dimension at axis 0.
        # We need to reshape it to match `final_target_shape`.
        # This involves adding `1` dimensions at the correct positions (if `hp_axis_pos != 0`)
        # and then broadcasting.

        # # Safest way: expand the current HP leaf to have 1s for all *other* batching dimensions
        # # at their target positions, and then broadcast.
        # temp_item = jnp.expand_dims(hp_leaf_with_hp_dim, axis=0)  # Add a dummy leading dim

        # Now, shift the HP dimension to its `hp_axis_pos` if it's not already there
        # and add 1s for other dimensions.
        # For this, we need to create a template of `1`s for all batching dimensions,
        # place the `hp_leaf_with_hp_dim` into that template, and then broadcast.

        # Let's consider the `hp_leaf_with_hp_dim` itself. It has shape (NumHPs, F1, F2...).
        # We want to embed this `NumHPs` at `hp_axis_pos`.
        # Create a shape (1,1,NumHPs,1,...) where NumHPs is at hp_axis_pos, other batch dims are 1.
        intermediate_shape_for_hp = [1] * (max_axis_index + 1)
        intermediate_shape_for_hp[hp_axis_pos] = num_hps
        intermediate_shape_for_hp = tuple(intermediate_shape_for_hp) + original_feature_dims

        # Reshape hp_leaf_with_hp_dim into this intermediate shape.
        # This requires `hp_leaf_with_hp_dim` to be first expanded to cover `original_feature_dims`
        # and then further expanded to match the `1`s.
        # The trick is that `jnp.reshape` (or explicit `expand_dims`) can be used if `item_leaf.size` matches the product of dims.

        # Simpler approach:
        # 1. Start with the hp_leaf (NumHPs, Features...)
        # 2. Flatten the features: (NumHPs, TotalFeatures)
        # 3. Create a structure of 1s for the target batch dimensions.
        #    This is `jnp.zeros(final_target_shape, dtype=hp_leaf_with_hp_dim.dtype)` if we need to copy.
        #    But we want views.

        # Let's use `_distribute_single_item`'s logic here, but with a special handling for the HP dim.
        # The `hp_leaf_with_hp_dim` is *not* a scalar; it's already (NumHPs, ...).
        # We need to treat (NumHPs) as its "core" value and then broadcast the rest.

        # This is where the complexity comes. The `_distribute_single_item` assumes a scalar or (features,)
        # and builds the batch dimensions around it.
        # Here, one of the batch dimensions (HP) is *already* part of the "core" data.

        # Redoing `_distribute_hp_leaf` more carefully for this specific case:
        # The input `hp_leaf_with_hp_dim` has its "HP" dimension as the first dimension.
        # We need to insert dimensions for all *other* axes (`seed`, `device`, `update_batch`)
        # at their respective `in_axes` positions, relative to the HP dimension's current position.

        # Let's find the current axis of the HP dimension. Assume it's 0.
        # And the target axis is `hp_axis_pos`.

        # 1. Expand `hp_leaf_with_hp_dim` with `1`s for all axes *before* the target `hp_axis_pos`.
        current_data = hp_leaf_with_hp_dim
        for _ in range(hp_axis_pos):
            current_data = jnp.expand_dims(current_data, axis=0)

        # 2. Expand `hp_leaf_with_hp_dim` with `1`s for all axes *after* the target `hp_axis_pos`.
        # The effective index for new dims depends on how many we've already added.
        for axis_idx_in_strategy in range(hp_axis_pos + 1, len(strategy.axes)):
            # The actual index where we need to add '1' is `axis_idx_in_strategy` plus `num_leading_ones_added_so_far`
            # which equals `axis_idx_in_strategy`.
            current_data = jnp.expand_dims(
                current_data, axis=axis_idx_in_strategy
            )  # Add at the correct logical position

        # Now `current_data` has shape like `(1,1,NumHPs,1,Features...)` if hp_axis_pos was 2.
        # Or `(NumHPs,1,1,1,Features...)` if hp_axis_pos was 0.

        # 3. Broadcast this expanded array to the final target shape calculated earlier.
        # The `final_target_shape` implicitly has `num_hps` at `hp_axis_pos`.
        final_target_shape_list = [1] * (max_axis_index + 1)
        for axis_spec in strategy.axes:
            final_target_shape_list[axis_spec.in_axes] = axis_spec.size
        final_target_shape = tuple(final_target_shape_list) + original_feature_dims

        return jnp.broadcast_to(current_data, final_target_shape)

    return jax.tree_util.tree_map(_distribute_hp_leaf, hyperparams_struct)


def distribute_env_states_and_timesteps_across_axes(
    env: Any,
    strategy: DistributionStrategy,
    key: jax.random.PRNGKey,
    max_num_envs_per_core: int,  # This is the 'envs' dim *per final core*
) -> tuple[Any, Any, jax.random.PRNGKey]:
    """
    Initializes environment states and timesteps, distributing them across the strategy axes.
    """
    # 1. Calculate the total number of environment resets needed.
    # This is the product of all strategy axis sizes multiplied by max_num_envs_per_core.
    total_env_resets_needed = max_num_envs_per_core
    for axis_spec in strategy.axes:
        total_env_resets_needed *= axis_spec.size

    # 2. Split keys for environment resets.
    key, *env_reset_keys_list = jax.random.split(key, total_env_resets_needed + 1)
    env_reset_keys = jnp.stack(env_reset_keys_list)  # (total_env_resets_needed, 2)

    # 3. Vmap env.reset over the flattened keys.
    env_states_flat, timesteps_flat = jax.vmap(env.reset, in_axes=(0))(env_reset_keys)

    # 4. Reshape the flattened env states and timesteps to match the strategy's dimensions
    # and the max_num_envs_per_core.
    # The target shape will be (Axis0_size, Axis1_size, ..., AxisN_size, max_num_envs_per_core, OriginalEnvStateDims...)
    max_axis_index = max((axis.in_axes for axis in strategy.axes), default=-1)
    target_prefix_shape_list = [1] * (max_axis_index + 1)  # Placeholder for batching dims

    for axis in strategy.axes:
        target_prefix_shape_list[axis.in_axes] = axis.size

    final_shape_for_env_data = tuple(target_prefix_shape_list) + (max_num_envs_per_core,)

    def reshape_env_data_fn(x: jnp.ndarray) -> jnp.ndarray:
        # x is (total_env_resets_needed, OriginalEnvStateDims...)
        return x.reshape(final_shape_for_env_data + x.shape[1:])

    env_states = jax.tree_util.tree_map(reshape_env_data_fn, env_states_flat)
    timesteps = jax.tree_util.tree_map(reshape_env_data_fn, timesteps_flat)

    return env_states, timesteps, key


def distribute_keys_across_axes(
    key: jax.random.PRNGKey,
    strategy: DistributionStrategy,
) -> tuple[Any, jax.random.PRNGKey]:
    """
    Distributes a base PRNGKey across the specified strategy axes.
    The final key shape will match the ordering implied by the strategy's in_axes.
    """
    # 1. Calculate the total number of independent PRNGKey pairs needed.
    # This is the product of all strategy axis sizes.
    total_num_key_pairs = 1
    for axis_spec in strategy.axes:
        total_num_key_pairs *= axis_spec.size

    # 2. Split the base key into the required number of individual keys.
    new_base_key, *individual_keys = jax.random.split(key, total_num_key_pairs + 1)

    # Stack them. Each element of `individual_keys` is already a (2,) array.
    # So `stacked_keys` will be (total_num_key_pairs, 2).
    stacked_keys = jnp.stack(individual_keys)

    # 3. Reshape the stacked keys to match the strategy's dimensions.
    # The target shape for the keys' leading dimensions.
    max_axis_index = max((axis.in_axes for axis in strategy.axes), default=-1)
    target_prefix_shape_list = [1] * (max_axis_index + 1)  # Placeholder for batching dims

    for axis in strategy.axes:
        target_prefix_shape_list[axis.in_axes] = axis.size

    final_key_shape = tuple(target_prefix_shape_list) + (
        2,
    )  # Add the (2,) for PRNGKey internal structure

    reshaped_keys = stacked_keys.reshape(final_key_shape)

    return reshaped_keys, new_base_key


def distribute_keys_across_axes_wo_update_batch_dim(
    key: jax.random.PRNGKey,
    strategy: DistributionStrategy,
) -> tuple[Any, jax.random.PRNGKey]:
    """
    Distributes a base PRNGKey across the specified strategy axes, *excluding* the 'update_batch' dimension.
    """
    # Create a modified strategy that excludes the 'update_batch' axis
    filtered_axes = tuple(axis for axis in strategy.axes if axis.name != "update_batch")
    filtered_strategy = DistributionStrategy(axes=filtered_axes)

    # Reuse the general function with the filtered strategy
    return distribute_keys_across_axes(key, filtered_strategy)


def broadcast_hp_batched_array_to_strategy_shape(
    hp_batched_array: jnp.ndarray,  # Expected shape (NumHPs, ...optional_feature_dims)
    strategy: DistributionStrategy,
    hyperparam_axis_name_in_strategy: str = "hyperparam",
) -> jnp.ndarray:
    """
    Takes an array already batched along its first dimension by hyperparameters,
    and broadcasts/replicates it to match the full distributed shape defined by strategy.

    Args:
        hp_batched_array: Array with leading HP dimension.
        strategy: The target distribution strategy.
        hyperparam_axis_name_in_strategy: Name of the hyperparameter axis in the strategy.

    Returns:
        The array broadcasted to the strategy's shape.
    """
    if hp_batched_array.ndim == 0:  # Scalar passed, not HP-batched
        # This case might imply a global scalar for all HPs; use distribute_initial_components
        raise ValueError(
            "Input array must be at least 1D (HP-batched). For global scalars, use distribute_initial_components."
        )

    num_hps_in_array = hp_batched_array.shape[0]
    feature_dims = hp_batched_array.shape[1:]

    # Determine target prefix shape (batching dimensions from strategy)
    max_axis_index = max((axis.in_axes for axis in strategy.axes), default=-1)
    target_prefix_shape_list = [1] * (max_axis_index + 1)
    found_hp_axis_in_strategy = False
    for axis in strategy.axes:
        if axis.name == hyperparam_axis_name_in_strategy:
            if axis.size != num_hps_in_array:
                raise ValueError(
                    f"Mismatch: strategy's hyperparam axis size ({axis.size}) "
                    f"does not match input array's HP dim size ({num_hps_in_array})."
                )
            target_prefix_shape_list[axis.in_axes] = axis.size
            found_hp_axis_in_strategy = True
        else:
            target_prefix_shape_list[axis.in_axes] = axis.size

    if not found_hp_axis_in_strategy:
        raise ValueError(
            f"Hyperparameter axis '{hyperparam_axis_name_in_strategy}' not found in strategy."
        )

    final_target_shape = tuple(target_prefix_shape_list) + feature_dims

    # Reshape hp_batched_array to align its HP dimension and add new axes for others
    hp_axis_pos_in_strategy = strategy.get_axis_position(hyperparam_axis_name_in_strategy)

    # # Start with the original array (NumHPs, Features...)
    # current_data_shape = list(hp_batched_array.shape)
    # Target shape for reshape before broadcast: (1, ..., NumHPs, ..., 1, Features...)
    # where NumHPs is at hp_axis_pos_in_strategy relative to other batching dims.
    reshape_target = [1] * len(target_prefix_shape_list)  # Only for batching dims
    reshape_target[hp_axis_pos_in_strategy] = num_hps_in_array

    # Combine with feature dimensions for the full reshape target
    full_reshape_target = tuple(reshape_target) + feature_dims

    value_to_broadcast = jnp.reshape(hp_batched_array, full_reshape_target)

    return jnp.broadcast_to(value_to_broadcast, final_target_shape)


def build_generic_distributed_state(
    learner_state_constructor: Callable,
    initial_components: dict[str, Any],
    hyperparam_structs: dict[str, Any],
    env: Any,
    key: chex.PRNGKey,
    train_strategy: DistributionStrategy,
    max_num_envs_per_core: int,
) -> Any:
    """
    Generic function to create a fully distributed learner state.

    Args:
        learner_state_constructor: The dataclass/NamedTuple constructor for the final learner state.
        initial_components: A dictionary of non-hyperparameter components to be distributed (e.g., {'params': ..., 'opt_states': ...}).
        hyperparam_structs: A dictionary of hyperparameter structs to be distributed.
        env: The environment instance.
        key: The base JAX PRNGKey.
        train_strategy: The distribution strategy.
        max_num_envs_per_core: The maximum number of environments per core for padding.

    Returns:
        An instance of the learner state, fully distributed across all axes.
    """
    # 1. Distribute environment states, timesteps, and keys
    env_states_dist, timesteps_dist, key_after_envs = (
        distribute_env_states_and_timesteps_across_axes(
            env, train_strategy, key, max_num_envs_per_core
        )
    )
    keys_dist, _ = distribute_keys_across_axes(key_after_envs, train_strategy)

    # 2. Distribute initial components (params, opt_states, norm_params, buffer_state, etc.)
    distributed_components = distribute_initial_components(initial_components, train_strategy)

    # print("hyperparam_structs:")
    # print(hyperparam_structs)
    # 3. Distribute all hyperparameter structs,
    # which are already correctly named by the caller
    distributed_hyperparams = {
        key: distribute_hyperparam_struct_across_axes(s, train_strategy, "hyperparam")
        for key, s in hyperparam_structs.items()
    }

    # 4. Construct the final learner state object
    return learner_state_constructor(
        **distributed_components,
        **distributed_hyperparams,
        key=keys_dist,
        env_state=env_states_dist,
        timestep=timesteps_dist,
    )
