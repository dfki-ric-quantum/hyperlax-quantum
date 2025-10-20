import logging
from dataclasses import dataclass
from typing import Any

import jax
import jax.numpy as jnp
from tabulate import tabulate

from hyperlax.layout.axes import AxisSpec, DistributionStrategy

logger = logging.getLogger(__name__)


@dataclass
class TransformationSummary:
    """Summary of all transformations applied to functions and data."""

    strategy_name: str
    axes_info: list[dict[str, Any]]
    function_transforms: list[str]
    state_operations: list[dict[str, Any]]
    final_shapes: dict[str, dict[str, Any]]
    device_info: dict[str, Any]


def analyze_layout_strategy(
    strategy: DistributionStrategy,
    sample_data: dict[str, Any],
    strategy_name: str = "Unknown",
    devices: list = None,
) -> TransformationSummary:
    """Analyze and summarize what the layout strategy does."""

    if devices is None:
        devices = jax.devices()

    # Analyze axes
    axes_info = []
    for i, axis in enumerate(strategy.axes):
        axis_info = {
            "position": i,
            "name": axis.name,
            "size": axis.size,
            "method": axis.method,
            "axis_name": axis.axis_name,
            "description": _get_axis_description(axis),
        }
        axes_info.append(axis_info)

    # Analyze function transformations
    function_transforms = []
    for axis in reversed(strategy.axes):  # Applied in reverse order
        transform_desc = _get_transform_description(axis)
        function_transforms.append(transform_desc)

    # Analyze state operations
    state_operations = []
    for i, axis in enumerate(strategy.axes):
        op_info = {
            "axis_position": i,
            "axis_name": axis.name,
            "operation": _get_state_operation_description(axis),
            "shape_change": f"Insert dim {i} with size {axis.size}",
        }
        state_operations.append(op_info)

    # Calculate final shapes
    final_shapes = {}
    for name, data in sample_data.items():
        original_shape = data.shape if hasattr(data, "shape") else (1,)  # Ensure data has shape
        final_shape_val = _calculate_final_shape(original_shape, strategy)
        original_size = jnp.prod(jnp.array(original_shape)).item() if original_shape else 1
        final_size = jnp.prod(jnp.array(final_shape_val)).item() if final_shape_val else 1

        final_shapes[name] = {
            "original": original_shape,
            "final": final_shape_val,
            "size_multiplier": (final_size / original_size) if original_size > 0 else 0,
        }

    # Device info
    device_info = {
        "total_devices": len(devices),
        "devices_used": _count_devices_used(strategy),
        "pmap_axes": [axis.name for axis in strategy.axes if axis.method == "pmap"],
        "memory_replication": any(axis.method == "pmap" for axis in strategy.axes),
    }

    return TransformationSummary(
        strategy_name=strategy_name,
        axes_info=axes_info,
        function_transforms=function_transforms,
        state_operations=state_operations,
        final_shapes=final_shapes,
        device_info=device_info,
    )


def print_layout_summary(summary: TransformationSummary):
    """Print a comprehensive summary of the layout strategy."""

    logger.info(f"\n{'=' * 80}")
    logger.info(f"LAYOUT STRATEGY SUMMARY: {summary.strategy_name}")
    logger.info(f"{'=' * 80}")

    # Axes Information
    axes_header = "\n AXES CONFIGURATION:"
    axes_table_data = []
    for axis in summary.axes_info:
        axes_table_data.append(
            [
                axis["position"],
                axis["name"],
                axis["size"],
                axis["method"],
                axis["axis_name"],
                axis["description"],
            ]
        )
    axes_table_str = tabulate(
        axes_table_data,
        headers=["Pos", "Name", "Size", "Method", "JAX Axis", "Description"],
        tablefmt="simple",
    )
    logger.info(f"{axes_header}\n{axes_table_str}")

    # Function Transformations
    func_trans_header = "\n🔧 FUNCTION TRANSFORMATIONS (Applied in this order):"
    func_trans_lines = [
        f"  {i}. {transform}" for i, transform in enumerate(summary.function_transforms, 1)
    ]
    logger.info(f"{func_trans_header}\n" + "\n".join(func_trans_lines))

    # State Operations
    state_ops_header = "\n STATE/DATA OPERATIONS:"
    state_table_data = []
    for op in summary.state_operations:
        state_table_data.append(
            [op["axis_position"], op["axis_name"], op["operation"], op["shape_change"]]
        )
    state_table_str = tabulate(
        state_table_data,
        headers=["Pos", "Axis", "Operation", "Shape Change"],
        tablefmt="simple",
    )
    logger.info(f"{state_ops_header}\n{state_table_str}")

    # Shape Analysis
    shape_analysis_header = "\n SHAPE TRANSFORMATIONS:"
    shape_table_data = []
    for name, shapes in summary.final_shapes.items():
        shape_table_data.append(
            [
                name,
                str(shapes["original"]),
                str(shapes["final"]),
                f"{shapes['size_multiplier']:.1f}x",  # Format multiplier
            ]
        )
    shape_table_str = tabulate(
        shape_table_data,
        headers=["Data", "Original Shape", "Final Shape", "Size Factor"],
        tablefmt="simple",
    )
    logger.info(f"{shape_analysis_header}\n{shape_table_str}")

    # Device Information
    device_info_header = "\n DEVICE INFORMATION:"
    device_table_data = [
        ["Total Devices Available", summary.device_info["total_devices"]],
        ["Devices Used", summary.device_info["devices_used"]],
        ["PMAP Axes", ", ".join(summary.device_info["pmap_axes"]) or "None"],
        [
            "Memory Replication",
            "Yes" if summary.device_info["memory_replication"] else "No",
        ],
    ]
    device_table_str = tabulate(
        device_table_data, headers=["Property", "Value"], tablefmt="simple"
    )
    logger.info(f"{device_info_header}\n{device_table_str}")

    logger.info(f"\n{'=' * 80}")


def _get_axis_description(axis: AxisSpec) -> str:
    """Get human-readable description of what an axis represents."""
    descriptions = {
        "update_batch": "Parallel updates within single step",
        "device": "Physical computation devices",
        "seed": "Independent agent seeds/runs",
        "hyperparam": "Hyperparameter configurations",
        "episode": "Episode batches",
        "env": "Environment instances",
    }
    return descriptions.get(axis.name, f"Custom axis: {axis.name}")


def _get_transform_description(axis: AxisSpec) -> str:
    """Get description of function transformation."""
    if axis.method == "vmap":
        return f"vmap over '{axis.name}' (size={axis.size}, axis_name='{axis.axis_name}')"
    elif axis.method == "pmap":
        return f"pmap over '{axis.name}' (size={axis.size}, axis_name='{axis.axis_name}') - DEVICE PARALLEL"
    elif axis.method == "scan":
        return f"scan over '{axis.name}' (length={axis.size})"
    else:
        return f"Unknown transform: {axis.method}"


def _get_state_operation_description(axis: AxisSpec) -> str:
    """Get description of state operation."""
    if axis.method == "vmap":
        return "Broadcast/expand"
    elif axis.method == "pmap":
        return "Replicate across devices"
    elif axis.method == "scan":
        return "Prepare for scan"
    else:
        return "Unknown operation"


def _calculate_final_shape(
    original_shape: tuple[int, ...], strategy: DistributionStrategy
) -> tuple[int, ...]:
    """Calculate the final shape after all transformations."""
    final_shape = list(original_shape)

    # Insert dimensions for each axis
    for i, axis in enumerate(strategy.axes):
        final_shape.insert(i, axis.size)

    return tuple(final_shape)


def _count_devices_used(strategy: DistributionStrategy) -> int:
    """Count how many devices are actually used."""
    for axis in strategy.axes:
        if axis.method == "pmap":
            return axis.size
    return 1  # No pmap means single device


# High-level analysis functions
def analyze_layout(
    train_strategy: DistributionStrategy,
    eval_strategy: DistributionStrategy,
    sample_state_components: dict[str, Any],
) -> None:
    """Analyze both training and evaluation layout strategies."""

    logger.info(f"\n{'=' * 80}")
    logger.info(" LAYOUT SUMMARY")
    logger.info("=" * 80)

    # Analyze training strategy
    train_summary = analyze_layout_strategy(
        train_strategy, sample_state_components, "Training Strategy"
    )
    print_layout_summary(train_summary)

    # Analyze evaluation strategy
    eval_summary = analyze_layout_strategy(
        eval_strategy, sample_state_components, "Evaluation Strategy"
    )
    print_layout_summary(eval_summary)

    # Compare strategies
    comparison_header = "\n STRATEGY COMPARISON:"
    comparison_table_data = [
        ["Aspect", "Training", "Evaluation"],
        ["Total Axes", len(train_strategy.axes), len(eval_strategy.axes)],
        [
            "Uses PMAP",
            any(a.method == "pmap" for a in train_strategy.axes),
            any(a.method == "pmap" for a in eval_strategy.axes),
        ],
        [
            "Memory Replication",
            train_summary.device_info["memory_replication"],
            eval_summary.device_info["memory_replication"],
        ],
    ]
    comparison_table_str = tabulate(comparison_table_data, headers="firstrow", tablefmt="simple")
    logger.info(f"{comparison_header}\n{comparison_table_str}")


def trace_data_flow(
    data_name: str,
    original_shape: tuple[int, ...],
    strategy: DistributionStrategy,
    show_intermediate_steps: bool = True,
) -> None:
    """Trace how data flows through the layout pipeline."""

    logger.info(f"\n DATA FLOW TRACE: {data_name}")
    logger.info("-" * 60)

    current_shape = list(original_shape)
    logger.info(f"Initial shape: {tuple(current_shape)}")

    if show_intermediate_steps:
        for i, axis in enumerate(strategy.axes):
            if axis.method == "pmap":
                operation = f"Replicate across {axis.size} devices"
            else:
                operation = f"Insert & broadcast dim {i}"

            current_shape.insert(i, axis.size)
            logger.info(f"Step {i + 1} ({axis.name}): {operation} -> {tuple(current_shape)}")

    final_shape = tuple(current_shape)
    original_size = jnp.prod(jnp.array(original_shape)).item() if original_shape else 1
    final_size = jnp.prod(jnp.array(final_shape)).item() if final_shape else 1
    size_factor = (final_size / original_size) if original_size > 0 else 0

    logger.info(f"Final shape: {final_shape}")
    logger.info(f"Memory factor: {size_factor:.1f}x original size")


# Complete analysis wrapper
def summarize_layout(setup_result: dict[str, Any]) -> None:
    """Complete summary of distributed setup."""

    # Extract components
    train_strategy = setup_result["train_strategy"]
    eval_strategy = setup_result["eval_strategy"]
    initial_state = setup_result["initial_state"]  # This is the full learner state

    # Sample state components for shape analysis
    # Ensure that attributes exist before trying to access them
    sample_components = {}
    if hasattr(initial_state, "params"):
        if hasattr(initial_state.params, "actor_params"):  # For ActorCriticParams
            sample_components["params"] = getattr(
                initial_state.params, "actor_params", initial_state.params
            )
        else:  # For direct params like OnlineAndTarget
            sample_components["params"] = initial_state.params
    if hasattr(initial_state, "env_state"):
        sample_components["env_state"] = initial_state.env_state
    if hasattr(initial_state, "key"):
        sample_components["keys"] = initial_state.key
    if hasattr(initial_state, "hyperparams"):
        sample_components["hyperparams"] = initial_state.hyperparams

    # Run analysis
    analyze_layout(train_strategy, eval_strategy, sample_components)

    # Trace specific data flows
    logger.info("\n🔍 DETAILED DATA FLOW TRACES:")

    param_data_for_trace = None
    if hasattr(initial_state, "params"):
        if hasattr(initial_state.params, "actor_params"):  # PPO
            param_data_for_trace = initial_state.params.actor_params
        elif hasattr(initial_state.params, "online"):  # DQN
            param_data_for_trace = initial_state.params.online
        else:  # Other structures
            param_data_for_trace = initial_state.params

    if param_data_for_trace:
        # Get a leaf's shape (e.g., first leaf of actor_params or online Q-params)
        param_leaf_shape = jax.tree_util.tree_leaves(param_data_for_trace)[0].shape
        # Remove the distributed dimensions to get original shape
        # This assumes the number of axes in train_strategy matches the number of leading distributed dims
        num_dist_dims = len(train_strategy.axes)
        if len(param_leaf_shape) >= num_dist_dims:
            original_param_shape = param_leaf_shape[num_dist_dims:]
            trace_data_flow(
                "Network Parameters (e.g., Actor/Q-Online)",
                original_param_shape,
                train_strategy,
            )
        else:
            logger.warning(
                f"Could not determine original param shape for trace. Leaf shape: {param_leaf_shape}, Dist dims: {num_dist_dims}"
            )
    else:
        logger.warning(
            "Could not trace Network Parameters: 'params' attribute not found or structured unexpectedly."
        )

    # Trace environment state - requires a typical obs shape.
    # This is a placeholder. In a real scenario, you'd get this from env_spec.
    # For 'env_state', the structure varies greatly. Let's trace a typical observation part.
    if (
        hasattr(initial_state, "env_state")
        and hasattr(initial_state.env_state, "obs")
        and hasattr(initial_state.env_state.obs, "agent_view")
    ):
        original_env_obs_shape = initial_state.env_state.obs.agent_view.shape[
            len(train_strategy.axes) :
        ]  # Remove distributed dims
        trace_data_flow(
            "Environment Observation (agent_view)",
            original_env_obs_shape,
            train_strategy,
        )
    elif hasattr(initial_state, "last_timestep") and hasattr(
        initial_state.last_timestep.observation, "agent_view"
    ):  # For PPO Learner State like structures
        original_env_obs_shape = initial_state.last_timestep.observation.agent_view.shape[
            len(train_strategy.axes) :
        ]
        trace_data_flow(
            "Environment Observation (agent_view from last_timestep)",
            original_env_obs_shape,
            train_strategy,
        )
    else:
        logger.warning(
            "Could not trace Environment Observation: 'env_state.obs.agent_view' or 'last_timestep.observation.agent_view' not found."
        )

    logger.info("\n Summary complete!")
