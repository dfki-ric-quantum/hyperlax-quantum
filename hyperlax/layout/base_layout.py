from hyperlax.layout.axes import AxisSpec, DistributionStrategy


def create_train_strategy(
    num_update_batch: int,
    num_devices: int,
    num_seeds: int,
    num_hyperparams: int,
    jit_enabled: bool = True,
) -> DistributionStrategy:
    # Order: HP (axis 0), Seed (axis 1), UB (axis 2), Device (axis 3)
    strategy = DistributionStrategy(
        axes=(
            AxisSpec(
                "hyperparam",
                num_hyperparams,
                "vmap",
                in_axes=0,
                out_axes=0,
                axis_name="hyperparam",
            ),
            AxisSpec(
                "seed",
                num_seeds,
                "vmap",
                in_axes=1,
                out_axes=1,
                axis_name="independent_agent",
            ),
            AxisSpec(
                "device",
                num_devices,
                "pmap" if jit_enabled else "vmap",
                in_axes=2,
                out_axes=2,
                axis_name="device",
            ),
            # AxisSpec("update_batch", num_update_batch, "vmap", in_axes=3, out_axes=3, axis_name="update_batch"),
        )
    )
    # print("TRAIN DISTRIBUTION STRATEGY AXES:")
    # for axis in strategy.axes:
    #     print(f"  {axis.name}: size={axis.size}, method={axis.method}, axis_name={axis.axis_name}")
    return strategy


def create_eval_strategy(
    num_devices: int,
    num_seeds: int,
    num_hyperparams: int,
    jit_enabled: bool = True,
) -> DistributionStrategy:
    strategy = DistributionStrategy(
        axes=(
            AxisSpec(
                "hyperparam",
                num_hyperparams,
                "vmap",
                in_axes=0,
                out_axes=0,
                axis_name="hyperparam",
            ),
            AxisSpec(
                "seed",
                num_seeds,
                "vmap",
                in_axes=1,
                out_axes=1,
                axis_name="independent_agent",
            ),
            AxisSpec(
                "device",
                num_devices,
                "pmap" if jit_enabled else "vmap",
                in_axes=2,
                out_axes=2,
                axis_name="device",
            ),
        )
    )
    # print("EVAL DISTRIBUTION STRATEGY AXES:")
    # for axis in strategy.axes:
    #     print(f"  {axis.name}: size={axis.size}, method={axis.method}, axis_name={axis.axis_name}")
    return strategy


## Below tests if cobebase can handle dynamic shape orders and stay axis-aware, we change the order to be [Seed, HP, ...
# def create_train_strategy(
#     num_update_batch: int, num_devices: int, num_seeds: int, num_hyperparams: int, jit_enabled: bool = True
# ) -> DistributionStrategy:
#     strategy =  DistributionStrategy(axes=(
#         # AxisSpec("hyperparam", num_hyperparams, "vmap", in_axes=0, out_axes=0, axis_name="hyperparam"),
#         # AxisSpec("seed", num_seeds, "vmap", in_axes=1, out_axes=1, axis_name="independent_agent"),
#         AxisSpec("seed", num_seeds, "vmap", in_axes=0, out_axes=0, axis_name="independent_agent"),
#         AxisSpec("hyperparam", num_hyperparams, "vmap", in_axes=1, out_axes=1, axis_name="hyperparam"),
#         AxisSpec("device", num_devices, "pmap" if jit_enabled else "vmap", in_axes=2, out_axes=2, axis_name="device"),
#         AxisSpec("update_batch", num_update_batch, "vmap", in_axes=3, out_axes=3, axis_name="update_batch"),
#     ))
#     print("TRAIN DISTRIBUTION STRATEGY AXES:")
#     for axis in strategy.axes:
#         print(f"  {axis.name}: size={axis.size}, method={axis.method}, axis_name={axis.axis_name}")
#     return strategy

# def create_eval_strategy(
#     num_devices: int,
#     #num_update_batch: int,
#     num_seeds: int,
#     num_hyperparams: int,
#     jit_enabled: bool = True,
# ) -> DistributionStrategy:
#     strategy = DistributionStrategy(axes=(
#         #AxisSpec("hyperparam", num_hyperparams, "vmap", in_axes=0, out_axes=0, axis_name="hyperparam"),
#         #AxisSpec("seed", num_seeds, "vmap", in_axes=1, out_axes=1, axis_name="independent_agent"),
#         AxisSpec("seed", num_seeds, "vmap", in_axes=0, out_axes=0, axis_name="independent_agent"),
#         AxisSpec("hyperparam", num_hyperparams, "vmap", in_axes=1, out_axes=1, axis_name="hyperparam"),
#         AxisSpec("device", num_devices, "pmap" if jit_enabled else "vmap", in_axes=2, out_axes=2, axis_name="device")
#     ))
#     print("EVAL DISTRIBUTION STRATEGY AXES:")
#     for axis in strategy.axes:
#         print(f"  {axis.name}: size={axis.size}, method={axis.method}, axis_name={axis.axis_name}")
#     return strategy
