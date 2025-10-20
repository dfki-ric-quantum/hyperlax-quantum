from collections.abc import Callable
from dataclasses import dataclass, replace
from typing import Any

import chex
import jax
import jax.numpy as jnp
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
from hyperlax.configs.main_base import BaseExperimentConfig
from hyperlax.env.make_env import make as make_env
from hyperlax.layout.base_layout import create_train_strategy
from hyperlax.logger.jax_debug import JAXDebugFilter
from hyperlax.normalizer.running_stats import normalizer_setup
from hyperlax.runner.base_types import AlgorithmInterface
from hyperlax.runner.batch_utils import build_hyperparam_batch
from hyperlax.runner.launcher_utils import (
    _get_default_values_from_config,
    _get_ordered_vectorized_keys,
    build_main_experiment_config,
)
from hyperlax.utils.algo_setup import (
    build_networks_and_optimizers,
    instantiate_from_config,
)


@dataclass
class MockArgs:
    algo_and_network_config: str
    env_config: str
    run_length_modifier: str = "quick"


@dataclass
class AlgoTestSetup:
    config: BaseExperimentConfig
    interface: AlgorithmInterface
    env: Any
    key: chex.PRNGKey
    strategy: Any
    hyperparam_batch_wrappers: dict[str, Any]
    nets_and_opts: Any
    normalizer_fns: tuple[Callable, Callable, Callable]
    non_vec_hps: Any


def _create_hyperparam_samples(config: BaseExperimentConfig, num_samples: int) -> dict[str, list[Any]]:
    defaults = _get_default_values_from_config(config)
    samples = {key: [val] * num_samples for key, val in defaults.items()}

    if "algorithm.hyperparam.actor_lr" in samples:
        samples["algorithm.hyperparam.actor_lr"] = [1e-4, 2e-4]
    if "algorithm.network.actor_network.pre_torso.layer_sizes" in samples:
        samples["algorithm.network.actor_network.pre_torso.layer_sizes"] = [[32], [64]]

    sample_id_key = next((k for k in defaults if k.endswith(".sample_id")), None)
    if sample_id_key:
        samples[sample_id_key] = list(range(num_samples))

    return samples


def _setup_test_environment(algo_config_name: str, env_config_name: str, interface: AlgorithmInterface):
    # Deactivate jax.debug.print to avoid host callback errors in test environments
    JAXDebugFilter.get_instance().deactivate_to_noop()

    args = MockArgs(algo_and_network_config=algo_config_name, env_config=env_config_name)
    exp_info = build_main_experiment_config(args)
    config = exp_info.experiment_config

    num_hps, num_seeds, num_devices = 2, 2, 1
    config.training = replace(config.training, num_devices=num_devices)

    env, _ = make_env(config=config)
    key = jax.random.PRNGKey(42)
    keys = interface.key_setup_fn(key)
    main_key, _, *net_keys_list = keys

    strategy = create_train_strategy(
        num_hyperparams=num_hps, num_seeds=num_seeds, num_devices=num_devices, num_update_batch=1, jit_enabled=False
    )

    samples = _create_hyperparam_samples(config, num_hps)
    vec_keys_algo = _get_ordered_vectorized_keys(config.algorithm.hyperparam, "algorithm.hyperparam")
    hp_arrays = {
        "algo": [[s[key] for key in vec_keys_algo] for s in [dict(zip(samples, t, strict=False)) for t in zip(*samples.values(), strict=False)]]
    }

    batch_wrappers = {
        "algo": build_hyperparam_batch(
            jnp.array(hp_arrays["algo"]), interface.vectorized_hyperparams_cls._fields, config.algorithm.hyperparam
        )
    }

    network_setup = interface.build_network_setup_fn()
    # Keys are needed for both network specs and standalone param specs
    net_spec_names = list(network_setup.network_specs.keys()) + list(
        network_setup.param_specs.keys()
    )
    network_keys = dict(zip(net_spec_names, net_keys_list, strict=False))

    nets_and_opts = build_networks_and_optimizers(
        network_setup,
        config,
        env,
        network_keys,
        batch_wrappers,
        instantiate_from_config,
        interface.build_network_fn,
        interface.build_optimizer_fn,
        mode="give_me_fns_and_init_params",
    )

    # HACK: Manually add the alpha optimizer for SAC, mirroring the main training loop
    if "sac" in algo_config_name:
        if "alpha" in network_setup.optimizer_specs:
            alpha_optimizer = interface.build_optimizer_fn(network_setup.optimizer_specs["alpha"])
            # =nets_and_opts= is a NamedTuple, so we must use _replace
            new_optimizers = nets_and_opts.optimizers.copy()
            new_optimizers["alpha"] = alpha_optimizer
            nets_and_opts = nets_and_opts._replace(optimizers=new_optimizers)

    normalizer_fns = normalizer_setup(True, "running_meanstd", env.observation_spec())

    return AlgoTestSetup(
        config,
        interface,
        env,
        main_key,
        strategy,
        batch_wrappers,
        nets_and_opts,
        normalizer_fns,
        interface.non_vectorized_hyperparams_cls,
    )


@pytest.fixture
def ppo_setup(request):
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
    env = "gymnax.cartpole" if "drpqc" not in request.param else "gymnax.pendulum"
    return _setup_test_environment(request.param, env, interface)


@pytest.fixture
def sac_setup(request):
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
    return _setup_test_environment(request.param, "gymnax.pendulum", interface)


@pytest.fixture
def dqn_setup(request):
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
    return _setup_test_environment(request.param, "gymnax.cartpole", interface)


@pytest.fixture
def mock_launch_dependencies(mocker):
    """Mocks all major functions called by the launch scripts."""
    mocks = {
        "load_benchmark_config": mocker.patch("hyperlax.runner.launch.load_benchmark_config"),
        "is_experiment_complete": mocker.patch("hyperlax.runner.launch.is_experiment_complete"),
        "launch_sampling_sweep": mocker.patch("hyperlax.runner.launch.launch_sampling_sweep"),
        "launch_optuna_sweep": mocker.patch("hyperlax.runner.launch.launch_optuna_sweep"),
        "save_args_config_and_metadata": mocker.patch(
            "hyperlax.runner.launch.save_args_config_and_metadata"
        ),
        "launch_sample_generation": mocker.patch(
            "hyperlax.runner.launch.launch_sample_generation"
        ),
        "build_main_experiment_config": mocker.patch(
            "hyperlax.runner.launch.build_main_experiment_config"
        ),
    }
    return mocks


@pytest.fixture
def mock_launch_args():
    """Provides a mutable mock object for command-line arguments for pipeline tests."""
    @dataclass
    class MockLaunchArgs:
        algo_and_network_config: str = ""
        env_config: str = ""
        run_length_modifier: str = "quick"
    return MockLaunchArgs()
