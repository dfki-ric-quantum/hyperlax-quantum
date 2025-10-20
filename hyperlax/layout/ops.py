from collections.abc import Callable

import jax

from hyperlax.layout.axes import DistributionStrategy


def transform_function_by_strategy(
    core_fn: Callable, strategy: DistributionStrategy, jit_enabled: bool = True
) -> Callable:
    # print("Transforming function with strategy axes:")
    # for axis in strategy.axes:
    #     print(f"  {axis.method} over '{axis.name}' (size={axis.size}, axis_name={axis.axis_name})")
    transformed_fn = core_fn
    for axis in strategy.axes:
        if axis.method == "vmap":
            transformed_fn = jax.vmap(
                transformed_fn,
                in_axes=axis.in_axes,
                out_axes=axis.out_axes,
                axis_name=axis.axis_name,
            )
        elif axis.method == "pmap":
            if jit_enabled:
                transformed_fn = jax.pmap(
                    transformed_fn,
                    in_axes=axis.in_axes,
                    out_axes=axis.out_axes,
                    axis_name=axis.axis_name,
                )
            else:
                transformed_fn = jax.vmap(
                    transformed_fn,
                    in_axes=axis.in_axes,
                    out_axes=axis.out_axes,
                    axis_name=axis.axis_name,
                )
        else:
            raise ValueError(f"Unknown transformation method: {axis.method}")
    return transformed_fn
