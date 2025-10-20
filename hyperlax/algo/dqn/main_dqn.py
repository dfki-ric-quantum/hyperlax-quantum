import logging
from dataclasses import dataclass
from typing import Any

import tyro

from hyperlax.algo.dqn.config import DQNConfig
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
from hyperlax.configs.main_base import BaseExperimentConfig
from hyperlax.logger.console import configure_global_logging
from hyperlax.runner.base_types import AlgorithmInterface
from hyperlax.runner.experiment_runner import run_experiment
from hyperlax.utils.jax_utils import print_xla_env_vars

logger = logging.getLogger(__name__)


def main(
    config_run: BaseExperimentConfig,
) -> tuple[list[dict[str, float]], Any, dict[str, float]]:
    logger.info("DQN main function started.")
    print_xla_env_vars()

    if not config_run.training.hyperparam_batch_enabled:
        logger.info(
            "Single run mode detected. The runner will handle it by creating a dummy batch."
        )

    optuna_obj_names = getattr(
        config_run,
        "optuna_objective_names_for_runner",
        ["peak_performance", "final_performance"],
    )

    dqn_interface = AlgorithmInterface(
        vectorized_hyperparams_cls=DQNVectorizedHyperparams,
        non_vectorized_hyperparams_cls=DQNNonVecHyperparams,
        algo_setup_fns_factory=build_dqn_algo_setup_fns_for_phase_training,
        key_setup_fn=setup_dqn_keys,
        get_eval_act_fn_callback_for_algo=get_dqn_eval_act_fn,
        algorithm_name_prefix="DQN",
        # Builder functions for testing
        build_network_setup_fn=build_dqn_network_setup,
        build_network_fn=build_dqn_network,
        build_optimizer_fn=build_dqn_optimizer,
        build_update_step_fn=build_dqn_update_step_fn,
        build_distributed_layout_fn=build_dqn_distributed_layout,
    )

    return run_experiment(
        config=config_run,
        algo_interface=dqn_interface,
        optuna_objective_names=optuna_obj_names,
    )


if __name__ == "__main__":
    from hyperlax.configs.algo.dqn_mlp import get_base_config
    from hyperlax.configs.env.gymnax.cartpole import GymnaxCartPoleConfig
    from hyperlax.configs.modifiers.common_settings import apply_quick_test_settings
    from hyperlax.configs.modifiers.experiment_modes import apply_single_run_mode

    @dataclass
    class DQNExperimentConfigForCLI(BaseExperimentConfig):
        algorithm: DQNConfig

    base_cfg_instance = get_base_config()
    base_cfg_instance.env = GymnaxCartPoleConfig()

    final_default_instance = apply_single_run_mode(apply_quick_test_settings(base_cfg_instance))

    cfg_cli = tyro.cli(DQNExperimentConfigForCLI, default=final_default_instance)

    configure_global_logging(cfg_cli.logger, overall_log_prefix="DQN_MAIN")
    main(cfg_cli)
