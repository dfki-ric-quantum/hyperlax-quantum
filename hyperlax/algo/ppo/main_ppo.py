import logging
from dataclasses import dataclass

import tyro

from hyperlax.algo.ppo.config import PPOConfig
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
from hyperlax.configs.main_base import BaseExperimentConfig
from hyperlax.logger.console import configure_global_logging
from hyperlax.runner.base_types import AlgorithmInterface
from hyperlax.runner.experiment_runner import run_experiment
from hyperlax.utils.jax_utils import print_xla_env_vars

logger = logging.getLogger(__name__)


def main(
    config_run: BaseExperimentConfig,
) -> tuple[list[dict[str, float]], BaseExperimentConfig]:
    logger.info("PPO main function started.")
    print_xla_env_vars()

    if not config_run.training.hyperparam_batch_enabled:
        logger.info(
            "Single run mode detected. The runner will handle it by creating a single hp batch."
        )

    optuna_obj_names = getattr(
        config_run,
        "optuna_objective_names_for_runner",
        ["peak_performance", "final_performance"],
    )

    ppo_interface = AlgorithmInterface(
        vectorized_hyperparams_cls=PPOVectorizedHyperparams,
        non_vectorized_hyperparams_cls=PPONonVecHyperparams,
        algo_setup_fns_factory=build_ppo_algo_setup_fns_for_phase_training,
        key_setup_fn=setup_ppo_keys,
        get_eval_act_fn_callback_for_algo=get_ppo_eval_act_fn,
        algorithm_name_prefix="PPO",
        # Builder functions for testing
        build_network_setup_fn=build_ppo_network_setup,
        build_network_fn=build_ppo_network,
        build_optimizer_fn=build_ppo_optimizer,
        build_update_step_fn=build_ppo_update_step_fn,
        build_distributed_layout_fn=build_ppo_distributed_layout,
    )

    return run_experiment(
        config=config_run,
        algo_interface=ppo_interface,
        optuna_objective_names=optuna_obj_names,
    )


if __name__ == "__main__":
    from hyperlax.configs.algo.ppo_mlp import get_base_config
    from hyperlax.configs.env.gymnax.pendulum import GymnaxPendulumConfig
    from hyperlax.configs.modifiers.common_settings import apply_quick_test_settings
    from hyperlax.configs.modifiers.experiment_modes import apply_single_run_mode

    @dataclass
    class PPOExperimentConfigForCLI(BaseExperimentConfig):
        algorithm: PPOConfig

    base_cfg_instance = get_base_config()
    base_cfg_instance.env = GymnaxPendulumConfig()

    final_default_instance = apply_single_run_mode(apply_quick_test_settings(base_cfg_instance))

    cfg_cli = tyro.cli(PPOExperimentConfigForCLI, default=final_default_instance)

    configure_global_logging(cfg_cli.logger, overall_log_prefix="PPO_MAIN")
    main(cfg_cli)
