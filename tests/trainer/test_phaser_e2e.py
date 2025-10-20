from dataclasses import replace

import jax
import jax.numpy as jnp
import pytest

from hyperlax.base_types import AlgorithmGlobalSetupArgs
from hyperlax.trainer.phaser import run_training_w_phaser

def _run_phasing_test(algo_setup):
    """
    Runs an end-to-end test of the phaser logic for a given algorithm setup.
    1. Starts with 2 HPs.
    2. Runs until one HP is complete.
    3. The phaser should internally re-slice and re-JIT for the remaining HP.
    4. Runs again with the single remaining HP.
    5. Verifies that the final state is valid and parameters have changed.
    """
    config = algo_setup.config
    config.training = replace(config.training, total_timesteps=100, num_evaluation=2)  # Milestones at 50, 100

    # Force different step rates. HP0 is faster and needs 100/25 = 4 updates.
    # HP1 is slower and needs 100/10 = 10 updates.
    steps_per_update_list = [25, 10]
    num_hyperparams = algo_setup.hyperparam_batch_wrappers["algo"].shape[0]
    assert num_hyperparams == 2, "This test is designed for exactly 2 initial HPs."

    # --- Setup Phaser ---
    algo_interface = algo_setup.interface
    algo_setup_fns = algo_interface.algo_setup_fns_factory()

    global_args = AlgorithmGlobalSetupArgs(
        env=algo_setup.env,
        eval_env=algo_setup.env,  # Use same env for simplicity
        config=config,
        normalizer_fns=algo_setup.normalizer_fns,
        get_eval_act_fn_callback=algo_interface.get_eval_act_fn_callback_for_algo,
        algo_specific_keys=algo_interface.key_setup_fn(algo_setup.key),
    )

    # --- Run Phased Training ---
    result = run_training_w_phaser(
        target_total_steps=config.training.total_timesteps,
        num_evaluation_milestones=config.training.num_evaluation,
        initial_num_hyperparams=num_hyperparams,
        hp_steps_per_update=steps_per_update_list,
        algo_setup_fns=algo_setup_fns,
        initial_hyperparam_configs_for_algo=algo_setup.hyperparam_batch_wrappers,
        non_vec_hyperparams=algo_setup.non_vec_hps,
        global_args=global_args,
        include_final_master_state=True,  # We need the state to check it
    )

    # --- Assertions ---
    # 1. Check completion status: The phaser loop runs until ALL are complete.
    assert result.final_active_status == [False, False]

    # 2. Check final step counts: Both should reach the target.
    assert result.final_env_steps[0] >= 100
    assert result.final_env_steps[1] >= 100

    # 3. Check that re-phasing happened.
    # The first HP needs ~4 updates (segments, if scan_len=1). The second needs ~10.
    # If phasing works, the total segments must be > 4.
    assert result.execution_history.segments_executed > 4, (
        "The number of segments suggests that training stopped after the first HP finished."
    )

    # 4. Check the final learner state shape
    final_state = result.final_master_learner_state
    assert final_state is not None
    hp_axis_pos = algo_setup.strategy.get_axis_position("hyperparam")
    final_hp_dim = jax.tree_util.tree_leaves(final_state.params)[0].shape[hp_axis_pos]
    assert final_hp_dim == num_hyperparams, "Final master state should contain all original HPs"

    # 5. Check that parameters for BOTH HPs have changed
    initial_state = algo_interface.build_distributed_layout_fn(
        env=algo_setup.env,
        nets_and_opts=algo_setup.nets_and_opts,
        hyperparam_batch_wrappers=algo_setup.hyperparam_batch_wrappers,
        normalizer_fns=algo_setup.normalizer_fns,
        key=algo_setup.key,
        train_strategy=algo_setup.strategy,
        config=config,
    )

    # Check HP 0
    initial_params_hp0 = jax.tree_util.tree_map(lambda x: x[0], initial_state.params)
    final_params_hp0 = jax.tree_util.tree_map(lambda x: x[0], final_state.params)
    initial_params_flat, _ = jax.tree_util.tree_flatten(initial_params_hp0)
    final_params_flat, _ = jax.tree_util.tree_flatten(final_params_hp0)
    assert any(
        not jnp.allclose(p_initial, p_final) for p_initial, p_final in zip(initial_params_flat, final_params_flat, strict=False)
    ), "Parameters for the first HP (index 0) did not change."

    # Check HP 1
    initial_params_hp1 = jax.tree_util.tree_map(lambda x: x[1], initial_state.params)
    final_params_hp1 = jax.tree_util.tree_map(lambda x: x[1], final_state.params)
    initial_params_flat, _ = jax.tree_util.tree_flatten(initial_params_hp1)
    final_params_flat, _ = jax.tree_util.tree_flatten(final_params_hp1)
    assert any(
        not jnp.allclose(p_initial, p_final) for p_initial, p_final in zip(initial_params_flat, final_params_flat, strict=False)
    ), "Parameters for the second HP (index 1) did not change."


@pytest.mark.parametrize("ppo_setup", ["ppo_mlp"], indirect=True)
@pytest.mark.long
def test_phaser_e2e_ppo(ppo_setup):
    _run_phasing_test(ppo_setup)


@pytest.mark.parametrize("sac_setup", ["sac_mlp"], indirect=True)
@pytest.mark.long
def test_phaser_e2e_sac(sac_setup):
    _run_phasing_test(sac_setup)


@pytest.mark.parametrize("dqn_setup", ["dqn_mlp"], indirect=True)
@pytest.mark.long
def test_phaser_e2e_dqn(dqn_setup):
    _run_phasing_test(dqn_setup)
