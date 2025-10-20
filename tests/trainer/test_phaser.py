from typing import NamedTuple

import jax.numpy as jnp
import pytest

from hyperlax.base_types import HPRuntimeState
from hyperlax.layout.axes import AxisSpec, DistributionStrategy
from hyperlax.trainer.phaser import (
    EvaluationSegmentInfo,
    HPStaticConfig,
    _check_and_mark_completed_hps,
    _determine_dynamic_scan_length,
    _get_active_hp_info,
    _identify_hps_for_evaluation,
    _initialize_hp_runtime_states,
    _update_milestone_progress,
)

# --- Mocks and Helpers ---


class MockLearnerState(NamedTuple):
    total_env_steps_counter: jnp.ndarray


def create_mock_strategy(num_hps: int, num_seeds: int = 1) -> DistributionStrategy:
    """Creates a simple, consistent strategy for testing."""
    return DistributionStrategy(
        axes=(
            AxisSpec("hyperparam", num_hps, "vmap", in_axes=0),
            AxisSpec("seed", num_seeds, "vmap", in_axes=1),
        )
    )


# --- Tests for Helper Functions ---


def test_initialize_hp_runtime_states():
    strategy = create_mock_strategy(num_hps=3, num_seeds=2)
    # Shape: (3_hp, 2_seed). Steps for HP0=10, HP1=20, HP2=30
    steps_counter = jnp.array([[10, 10], [20, 20], [30, 30]])
    mock_state = MockLearnerState(total_env_steps_counter=steps_counter)

    runtime_states = _initialize_hp_runtime_states(3, mock_state, strategy)

    assert len(runtime_states) == 3
    assert all(rt.is_active for rt in runtime_states)
    assert all(rt.current_milestone_target_idx == 0 for rt in runtime_states)
    assert [rt.original_index for rt in runtime_states] == [0, 1, 2]
    # get_env_step_counter sums over all other dims, so we check the raw value.
    # It takes the 0th seed, so steps are [10, 20, 30]
    assert [rt.env_steps for rt in runtime_states] == [10, 20, 30]


def test_get_active_hp_info():
    states = [
        HPRuntimeState(original_index=0, env_steps=100, is_active=True, current_milestone_target_idx=1),
        HPRuntimeState(original_index=1, env_steps=200, is_active=False, current_milestone_target_idx=2),
        HPRuntimeState(original_index=2, env_steps=50, is_active=True, current_milestone_target_idx=0),
    ]
    active_rts, active_indices = _get_active_hp_info(states)

    assert len(active_rts) == 2
    assert [rt.original_index for rt in active_rts] == [0, 2]
    assert active_indices == [0, 2]


@pytest.mark.parametrize(
    "hp_states, milestones, expected_len",
    [
        # Basic case: min updates needed is ceil(100/10) = 10
        ([HPRuntimeState(0, 0, True, 0)], [100, 200], 10),
        # Multiple HPs: min is ceil(50/20) = 3 for HP1.
        ([HPRuntimeState(0, 0, True, 0), HPRuntimeState(1, 50, True, 0)], [100, 200], 3),
        # One HP is already at a milestone, should trigger eval -> scan_len=1
        ([HPRuntimeState(0, 100, True, 0), HPRuntimeState(1, 50, True, 0)], [100, 200], 1),
        # All HPs are done (milestone index is past the list)
        ([HPRuntimeState(0, 200, True, 2), HPRuntimeState(1, 200, True, 2)], [100, 200], 0),
        # One HP needs just a few steps, ceil(5/10) = 1
        ([HPRuntimeState(0, 95, True, 0)], [100, 200], 1),
    ],
)
def test_determine_dynamic_scan_length(hp_states, milestones, expected_len):
    static_configs = [
        HPStaticConfig(steps_per_update=10, original_index=0),
        HPStaticConfig(steps_per_update=20, original_index=1),
    ]
    scan_len = _determine_dynamic_scan_length(hp_states, static_configs, milestones)
    assert scan_len == expected_len


def test_identify_hps_for_evaluation():
    hp_rts = [
        HPRuntimeState(0, 100, True, 0),  # Exactly at milestone 0
        HPRuntimeState(1, 90, True, 0),  # Not yet at milestone 0
        HPRuntimeState(2, 150, True, 0),  # Overshot milestone 0
        HPRuntimeState(3, 200, True, 1),  # At milestone 1
        HPRuntimeState(4, 100, False, 0),  # Inactive, should be ignored
    ]
    active_indices = [0, 1, 2, 3]
    milestones = [100, 200]
    eval_info = _identify_hps_for_evaluation(hp_rts, active_indices, milestones)

    assert isinstance(eval_info, EvaluationSegmentInfo)
    # Indices are relative to the active_indices list
    assert eval_info.indices_to_eval == [0, 2, 3]
    # Original indices are the global ones
    assert eval_info.original_indices == [0, 2, 3]
    assert eval_info.milestone_values == [100, 100, 200]


def test_update_milestone_progress():
    hp_rts = [
        HPRuntimeState(0, 100, True, 0),
        HPRuntimeState(1, 90, True, 0),
        HPRuntimeState(2, 150, True, 1),
    ]
    eval_info = EvaluationSegmentInfo(indices_to_eval=[0, 2], original_indices=[0, 2], milestone_values=[100, 200])
    milestones = [100, 200, 300]

    updated_rts, history = _update_milestone_progress(hp_rts, eval_info, milestones)

    assert updated_rts[0].current_milestone_target_idx == 1
    assert updated_rts[1].current_milestone_target_idx == 0  # Unchanged
    assert updated_rts[2].current_milestone_target_idx == 2
    assert len(history) == 2
    assert history[0]["hp_original_idx"] == 0
    assert history[0]["milestone_value_hit"] == 100


def test_check_and_mark_completed_hps():
    hp_rts = [
        HPRuntimeState(0, 490, True, 4),
        HPRuntimeState(1, 500, True, 4),  # Exactly complete
        HPRuntimeState(2, 550, True, 4),  # Overshot
        HPRuntimeState(3, 600, False, 5),  # Already inactive
    ]
    target_steps = 500

    updated_rts, completed_indices = _check_and_mark_completed_hps(hp_rts, target_steps)

    assert updated_rts[0].is_active is True
    assert updated_rts[1].is_active is False
    assert updated_rts[2].is_active is False
    assert updated_rts[3].is_active is False  # Stays inactive
    assert completed_indices == [1, 2]
