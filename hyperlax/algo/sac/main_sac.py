import logging
from dataclasses import dataclass
from typing import Any

import tyro

from hyperlax.algo.sac.config import SACConfig
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
from hyperlax.logger.console import configure_global_logging
from hyperlax.runner.base_types import AlgorithmInterface
from hyperlax.runner.experiment_runner import run_experiment
from hyperlax.utils.jax_utils import print_xla_env_vars

logger = logging.getLogger(__name__)


def main(
    config_run: BaseExperimentConfig,
) -> tuple[list[dict[str, float]], Any, dict[str, float]]:
    logger.info("SAC main function started.")
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

    sac_interface = AlgorithmInterface(
        vectorized_hyperparams_cls=SACVectorizedHyperparams,
        non_vectorized_hyperparams_cls=SACNonVecHyperparams,
        algo_setup_fns_factory=build_sac_algo_setup_fns_for_phase_training,
        key_setup_fn=setup_sac_keys,
        get_eval_act_fn_callback_for_algo=get_sac_eval_act_fn,
        algorithm_name_prefix="SAC",
        # Builder functions for testing
        build_network_setup_fn=build_sac_network_setup,
        build_network_fn=build_sac_network,
        build_optimizer_fn=build_sac_optimizer,
        build_update_step_fn=build_sac_update_step_fn,
        build_distributed_layout_fn=build_sac_distributed_layout,
    )

    return run_experiment(
        config=config_run,
        algo_interface=sac_interface,
        optuna_objective_names=optuna_obj_names,
    )


if __name__ == "__main__":
    from hyperlax.configs.algo.sac_mlp import get_base_config
    from hyperlax.configs.env.gymnax.pendulum import GymnaxPendulumConfig
    from hyperlax.configs.modifiers.common_settings import apply_quick_test_settings
    from hyperlax.configs.modifiers.experiment_modes import apply_single_run_mode

    @dataclass
    class SACExperimentConfigForCLI(BaseExperimentConfig):
        algorithm: SACConfig

    base_cfg_instance = get_base_config()
    base_cfg_instance.env = GymnaxPendulumConfig()
    final_default_instance = apply_single_run_mode(apply_quick_test_settings(base_cfg_instance))

    cfg_cli = tyro.cli(SACExperimentConfigForCLI, default=final_default_instance)

    configure_global_logging(cfg_cli.logger, overall_log_prefix="SAC_MAIN")
    main(cfg_cli)
