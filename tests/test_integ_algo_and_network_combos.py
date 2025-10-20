import dataclasses
from typing import Any

import jax
import jax.numpy as jnp
import numpy as np
import pytest
from hyperlax.algo.dqn.setup_dqn import (
    build_dqn_algo_setup_fns_for_phase_training,
    build_dqn_distributed_layout,
    build_dqn_network,
    build_dqn_network_setup,
    build_dqn_optimizer,
    build_dqn_update_step_fn,
    get_dqn_eval_act_fn,
    setup_dqn_keys,
)
from hyperlax.algo.dqn.struct_dqn import DQNNonVecHyperparams, DQNVectorizedHyperparams
from hyperlax.algo.ppo.setup_ppo import (
    build_ppo_algo_setup_fns_for_phase_training,
    build_ppo_distributed_layout,
    build_ppo_network,
    build_ppo_network_setup,
    build_ppo_optimizer,
    build_ppo_update_step_fn,
    get_ppo_eval_act_fn,
    setup_ppo_keys,
)
from hyperlax.algo.ppo.struct_ppo import PPONonVecHyperparams, PPOVectorizedHyperparams
from hyperlax.algo.sac.setup_sac import (
    build_sac_algo_setup_fns_for_phase_training,
    build_sac_distributed_layout,
    build_sac_network,
    build_sac_network_setup,
    build_sac_optimizer,
    build_sac_update_step_fn,
    get_sac_eval_act_fn,
    setup_sac_keys,
)
from hyperlax.algo.sac.struct_sac import SACNonVecHyperparams, SACVectorizedHyperparams
from hyperlax.base_types import AlgorithmGlobalSetupArgs
from hyperlax.configs.main_base import BaseExperimentConfig
from hyperlax.configs.modifiers.common_settings import apply_quick_test_settings
from hyperlax.env.make_env import make as make_env
from hyperlax.hyperparam.base_types import flatten_tunables
from hyperlax.hyperparam.distributions import LogUniform, UniformContinuous, UniformDiscrete, Categorical
from hyperlax.network.hyperparam import MLPVectorizedHyperparams
from hyperlax.network.parametric_torso import ACTIVATION_FN_TO_IDX
from hyperlax.normalizer.running_stats import normalizer_setup
from hyperlax.runner.base_types import AlgorithmInterface
from hyperlax.runner.batch_utils import build_hyperparam_batch
from hyperlax.runner.launcher_utils import _get_ordered_vectorized_keys, build_main_experiment_config
from hyperlax.runner.sampling import _get_default_values_from_config
from hyperlax.utils.type_cast import cast_value_to_expected_type


def create_hyperparam_samples(config: BaseExperimentConfig, num_samples: int) -> dict[str, list[Any]]:
    """
    Creates a dictionary of distinct hyperparameter samples for testing.

    It finds the first non-fixed, vectorized parameter and generates `num_samples`
    distinct values for it based on its distribution. All other parameters
    are set to their default values.
    """
    defaults = _get_default_values_from_config(config)
    samples = {key: [val] * num_samples for key, val in defaults.items()}

    # Find the first tunable, non-fixed, vectorized parameter to create variation
    key_to_vary = None
    spec_to_vary = None
    flat_tunables = flatten_tunables(config.algorithm)
    for path, spec in flat_tunables.items():
        if spec.is_vectorized and not spec.is_fixed and spec.distribution:
            key_to_vary = f"algorithm.{path}"
            spec_to_vary = spec
            break

    if key_to_vary and num_samples > 1:
        print(f"Varying hyperparameter '{key_to_vary}' for test samples.")
        dist = spec_to_vary.distribution
        new_values = []
        if isinstance(dist, (UniformContinuous, LogUniform)):
            new_values = np.linspace(dist.domain[0], dist.domain[1], num_samples).tolist()
        elif isinstance(dist, UniformDiscrete):
            new_values = np.linspace(dist.domain[0], dist.domain[1], num_samples).astype(int).tolist()
        elif isinstance(dist, Categorical):
            # Take the first N choices, repeating the last if necessary
            choices = dist.values
            new_values = (choices * num_samples)[:num_samples]
        else:
            print(f"Warning: Cannot create varied samples for distribution type {type(dist)}. " "Using default values.")

        if new_values:
            # Ensure the list has the correct length
            if len(new_values) > num_samples:
                new_values = new_values[:num_samples]
            elif len(new_values) < num_samples:
                new_values.extend([new_values[-1]] * (num_samples - len(new_values)))
            samples[key_to_vary] = new_values

    sample_id_key = next((k for k in defaults if k.endswith(".sample_id")), None)
    if sample_id_key:
        samples[sample_id_key] = list(range(num_samples))

    return samples


@pytest.mark.parametrize(
    "algo_and_network_config",
    [
        # PPO
        "ppo_mlp",
        "ppo_drpqc",
        "ppo_tmlp",
        "ppo_vec_mlp",
        # SAC
        "sac_mlp",
        "sac_drpqc",
        "sac_tmlp",
        # DQN
        "dqn_mlp",
        "dqn_drpqc",
        "dqn_tmlp",
    ],
)
@pytest.mark.long
def test_config_build_and_single_step(algo_and_network_config, mock_launch_args):
    """
    Tests the full pipeline from config name to a single successful, jitted training step.
    This validates that config loading, environment patching, and algorithm setup
    are correctly wired for every supported combination.
    """
    # --- 1. Determine Environment and Algorithm Interface ---
    if "sac" in algo_and_network_config or "ppo" in algo_and_network_config:
        mock_launch_args.env_config = "gymnax.pendulum"  # Continuous action space
    else:  # DQN
        mock_launch_args.env_config = "gymnax.cartpole"  # Discrete action space

    mock_launch_args.algo_and_network_config = algo_and_network_config

    if "ppo" in algo_and_network_config:
        interface = AlgorithmInterface(
            vectorized_hyperparams_cls=PPOVectorizedHyperparams,
            non_vectorized_hyperparams_cls=PPONonVecHyperparams,
            algo_setup_fns_factory=build_ppo_algo_setup_fns_for_phase_training,
            key_setup_fn=setup_ppo_keys,
            get_eval_act_fn_callback_for_algo=get_ppo_eval_act_fn,
            algorithm_name_prefix="PPO",
            build_network_setup_fn=build_ppo_network_setup,
            build_network_fn=build_ppo_network,
            build_optimizer_fn=build_ppo_optimizer,
            build_update_step_fn=build_ppo_update_step_fn,
            build_distributed_layout_fn=build_ppo_distributed_layout,
        )
    elif "sac" in algo_and_network_config:
        interface = AlgorithmInterface(
            vectorized_hyperparams_cls=SACVectorizedHyperparams,
            non_vectorized_hyperparams_cls=SACNonVecHyperparams,
            algo_setup_fns_factory=build_sac_algo_setup_fns_for_phase_training,
            key_setup_fn=setup_sac_keys,
            get_eval_act_fn_callback_for_algo=get_sac_eval_act_fn,
            algorithm_name_prefix="SAC",
            build_network_setup_fn=build_sac_network_setup,
            build_network_fn=build_sac_network,
            build_optimizer_fn=build_sac_optimizer,
            build_update_step_fn=build_sac_update_step_fn,
            build_distributed_layout_fn=build_sac_distributed_layout,
        )
    elif "dqn" in algo_and_network_config:
        interface = AlgorithmInterface(
            vectorized_hyperparams_cls=DQNVectorizedHyperparams,
            non_vectorized_hyperparams_cls=DQNNonVecHyperparams,
            algo_setup_fns_factory=build_dqn_algo_setup_fns_for_phase_training,
            key_setup_fn=setup_dqn_keys,
            get_eval_act_fn_callback_for_algo=get_dqn_eval_act_fn,
            algorithm_name_prefix="DQN",
            build_network_setup_fn=build_dqn_network_setup,
            build_network_fn=build_dqn_network,
            build_optimizer_fn=build_dqn_optimizer,
            build_update_step_fn=build_dqn_update_step_fn,
            build_distributed_layout_fn=build_dqn_distributed_layout,
        )
    else:
        raise ValueError(f"Unknown algorithm in config name: {algo_and_network_config}")

    # --- 2. Build Config and Environment ---
    exp_info = build_main_experiment_config(mock_launch_args)
    config = apply_quick_test_settings(exp_info.experiment_config)
    env, eval_env = make_env(config=config)

    # --- 3. Setup Generic Learner Components ---
    num_hps, num_seeds, num_devices = 2, 2, 1
    config.training = dataclasses.replace(config.training, num_devices=num_devices, num_agents_slash_seeds=num_seeds)

    key = jax.random.PRNGKey(42)
    algo_keys = interface.key_setup_fn(key)

    # Generate distinct samples using the robust utility
    samples = create_hyperparam_samples(config, num_hps)
    hp_arrays = {}

    # Get all tunable specs to know their expected types for casting
    flat_tunables = flatten_tunables(config)

    # Helper to create array data from samples dict, casting values correctly
    def create_array_for_component(component_config, prefix):
        vec_keys = _get_ordered_vectorized_keys(component_config, prefix)
        if not vec_keys:
            return None

        rows = []
        for i in range(num_hps):
            row = []
            for key in vec_keys:
                raw_value = samples[key][i]
                # Special handling for vectorized activation, which must be an int index
                if "vec_mlp" in algo_and_network_config and key.endswith(".activation"):
                    casted_value = ACTIVATION_FN_TO_IDX[raw_value]
                else:
                    expected_type = flat_tunables[key].expected_type
                    casted_value = cast_value_to_expected_type(raw_value, expected_type)
                row.append(casted_value)
            rows.append(row)
        return rows

    hp_arrays["algo"] = create_array_for_component(config.algorithm.hyperparam, "algorithm.hyperparam")
    if hasattr(config.algorithm.network, "actor_network"):
        actor_array = create_array_for_component(
            config.algorithm.network.actor_network, "algorithm.network.actor_network"
        )
        if actor_array:
            hp_arrays["network_actor"] = actor_array
    if hasattr(config.algorithm.network, "critic_network"):
        critic_array = create_array_for_component(
            config.algorithm.network.critic_network, "algorithm.network.critic_network"
        )
        if critic_array:
            hp_arrays["network_critic"] = critic_array

    config.training = dataclasses.replace(
        config.training,
        hyperparam_batch_enabled=True,
        hyperparam_batch_size=num_hps,
        hyperparam_batch_samples=hp_arrays,
        hyperparam_batch_sample_ids=list(range(num_hps)),
    )

    # --- Manually create the batch wrappers, mimicking the main runner ---
    batch_wrappers = {}
    hp_arrays_dict = config.training.hyperparam_batch_samples

    batch_wrappers["algo"] = build_hyperparam_batch(
        array=jnp.array(hp_arrays_dict["algo"]),
        expected_fields=interface.vectorized_hyperparams_cls._fields,
        base_config_component=config.algorithm.hyperparam,
    )
    if "network_actor" in hp_arrays_dict:
        batch_wrappers["network_actor"] = build_hyperparam_batch(
            array=jnp.array(hp_arrays_dict["network_actor"]),
            expected_fields=MLPVectorizedHyperparams._fields,
            base_config_component=config.algorithm.network.actor_network,
        )
    if "network_critic" in hp_arrays_dict:
        batch_wrappers["network_critic"] = build_hyperparam_batch(
            array=jnp.array(hp_arrays_dict["network_critic"]),
            expected_fields=MLPVectorizedHyperparams._fields,
            base_config_component=config.algorithm.network.critic_network,
        )

    global_args = AlgorithmGlobalSetupArgs(
        env=env,
        eval_env=eval_env,
        config=config,
        normalizer_fns=normalizer_setup(True, "running_meanstd", env.observation_spec()),
        get_eval_act_fn_callback=interface.get_eval_act_fn_callback_for_algo,
        algo_specific_keys=algo_keys,
    )

    setup_fns = interface.algo_setup_fns_factory()
    algo_setup_result = setup_fns.setup_initial(
        batch_wrappers,
        interface.non_vectorized_hyperparams_cls,
        global_args,
        1,  # num_updates_per_scan
    )

    # --- 4. Execute a single step ---
    initial_state = algo_setup_result.learner_state
    train_fn = algo_setup_result.train_one_unit_fn

    # The first call will JIT compile
    output = train_fn(initial_state)
    final_state = output.learner_state
    final_state.total_env_steps_counter.block_until_ready()

    # --- 5. Assertions ---
    assert final_state is not None
    assert jax.tree_util.tree_structure(initial_state) == jax.tree_util.tree_structure(final_state)

    # Check shapes are preserved
    initial_shapes = jax.tree_util.tree_leaves(jax.tree_util.tree_map(lambda x: x.shape, initial_state))
    final_shapes = jax.tree_util.tree_leaves(jax.tree_util.tree_map(lambda x: x.shape, final_state))
    assert initial_shapes == final_shapes

    # Check that parameters changed
    initial_params_flat, _ = jax.tree_util.tree_flatten(initial_state.params)
    final_params_flat, _ = jax.tree_util.tree_flatten(final_state.params)
    assert any(
        not jnp.allclose(p_initial, p_final)
        for p_initial, p_final in zip(initial_params_flat, final_params_flat, strict=False)
    ), "Parameters did not change after one update step."

    # Check that environment steps increased
    assert jnp.all(final_state.total_env_steps_counter > initial_state.total_env_steps_counter)
