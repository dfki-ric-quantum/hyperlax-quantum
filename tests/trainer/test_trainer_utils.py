import jax.numpy as jnp
import pytest

from hyperlax.layout.axes import AxisSpec, DistributionStrategy
from hyperlax.trainer.utils import _calculate_milestones, sum_total_env_steps_per_hyperparam


# --- Test _calculate_milestones ---
@pytest.mark.parametrize(
    "total_steps, num_milestones, expected",
    [
        (100, 4, [25, 50, 75, 100]),
        (100, 5, [20, 40, 60, 80, 100]),
        (99, 4, [24, 48, 72, 99]),
        (100, 0, [100]),
        (0, 5, [0]),
        (10, 10, [1, 2, 3, 4, 5, 6, 7, 8, 9, 10]),
        (10, 11, [1, 2, 3, 4, 5, 6, 7, 8, 9, 10]),  # More milestones than steps
        (100, 3, [33, 66, 100]),
    ],
)
def test_calculate_milestones(total_steps, num_milestones, expected):
    assert _calculate_milestones(total_steps, num_milestones) == expected


# --- Test sum_total_env_steps_per_hyperparam ---
def test_sum_total_env_steps_per_hyperparam_standard_layout():
    # Layout: (Seed, HP, Device, UpdateBatch) -> in_axes=(0, 1, 2, 3) (a common order)
    strategy = DistributionStrategy(
        axes=(
            AxisSpec("seed", 3, "vmap", in_axes=0, out_axes=0),
            AxisSpec("hyperparam", 2, "vmap", in_axes=1, out_axes=1),
            AxisSpec("device", 4, "pmap", in_axes=2, out_axes=2),
            AxisSpec("update_batch", 1, "vmap", in_axes=3, out_axes=3),
        )
    )

    # Shape: (3_seed, 2_hp, 4_device, 1_ub)
    # The values across seed and update_batch dims are identical, so we take the 0-th slice.
    # Then we sum across the device dimension.
    # For HP0: 10 + 11 + 12 + 13 = 46
    # For HP1: 20 + 21 + 22 + 23 = 86
    env_steps_counter = jnp.array(
        [
            [
                [[10], [11], [12], [13]],  # seed 0, hp 0
                [[20], [21], [22], [23]],
            ],  # seed 0, hp 1
            [
                [[10], [11], [12], [13]],  # seed 1, hp 0
                [[20], [21], [22], [23]],
            ],  # seed 1, hp 1
            [
                [[10], [11], [12], [13]],  # seed 2, hp 0
                [[20], [21], [22], [23]],
            ],  # seed 2, hp 1
        ]
    )

    result = sum_total_env_steps_per_hyperparam(env_steps_counter, strategy)

    assert result.shape == (2,)
    assert jnp.allclose(result, jnp.array([46, 86]))


def test_sum_total_env_steps_per_hyperparam_different_layout():
    # Layout: (HP, Device, Seed, UB) -> in_axes=(0, 1, 2, 3)
    strategy = DistributionStrategy(
        axes=(
            AxisSpec("hyperparam", 2, "vmap", in_axes=0, out_axes=0),
            AxisSpec("device", 3, "pmap", in_axes=1, out_axes=1),
            AxisSpec("seed", 4, "vmap", in_axes=2, out_axes=2),
            AxisSpec("update_batch", 1, "vmap", in_axes=3, out_axes=3),
        )
    )
    # Shape: (2_hp, 3_device, 4_seed, 1_ub)
    # We take the 0-th seed. Sum across devices.
    # For HP0: 10 + 11 + 12 = 33
    # For HP1: 20 + 21 + 22 = 63
    env_steps_counter = jnp.array(
        [
            [
                [[10], [10], [10], [10]],  # hp 0, device 0
                [[11], [11], [11], [11]],  # hp 0, device 1
                [[12], [12], [12], [12]],
            ],  # hp 0, device 2
            [
                [[20], [20], [20], [20]],  # hp 1, device 0
                [[21], [21], [21], [21]],  # hp 1, device 1
                [[22], [22], [22], [22]],
            ],  # hp 1, device 2
        ]
    )

    result = sum_total_env_steps_per_hyperparam(env_steps_counter, strategy)

    assert result.shape == (2,)
    assert jnp.allclose(result, jnp.array([33, 63]))


def test_sum_total_env_steps_with_update_batch():
    # Layout: (HP, UB, Device, Seed) -> in_axes=(0, 1, 2, 3)
    strategy = DistributionStrategy(
        axes=(
            AxisSpec("hyperparam", 2, "vmap", in_axes=0, out_axes=0),
            AxisSpec("update_batch", 2, "vmap", in_axes=1, out_axes=1),
            AxisSpec("device", 3, "pmap", in_axes=2, out_axes=2),
            AxisSpec("seed", 1, "vmap", in_axes=3, out_axes=3),
        )
    )
    # Shape: (2_hp, 2_ub, 3_device, 1_seed)
    # Take 0-th element from seed and update_batch axes.
    # Then sum over device.
    # For HP0: 10 + 11 + 12 = 33
    # For HP1: 20 + 21 + 22 = 63
    env_steps_counter = jnp.array(
        [
            [  # HP 0
                [  # UB 0
                    [10],
                    [11],
                    [12],
                ],
                [  # UB 1 (should be ignored)
                    [99],
                    [99],
                    [99],
                ],
            ],
            [  # HP 1
                [  # UB 0
                    [20],
                    [21],
                    [22],
                ],
                [  # UB 1 (should be ignored)
                    [99],
                    [99],
                    [99],
                ],
            ],
        ]
    )
    result = sum_total_env_steps_per_hyperparam(env_steps_counter, strategy)
    assert result.shape == (2,)
    assert jnp.allclose(result, jnp.array([33, 63]))
