from typing import Any

from hyperlax.algo.sac.config import SACConfig
from hyperlax.algo.sac.hyperparam import SACHyperparams
from hyperlax.configs.env.base_env import BaseEnvironmentConfig
from hyperlax.configs.logger.base_logger import LoggerConfig
from hyperlax.configs.main_base import BaseExperimentConfig
from hyperlax.configs.network.actorcritic_sac_tmlp import SACTMLPActorCriticConfig
from hyperlax.configs.training.base_training import BaseTrainingConfig
from hyperlax.hyperparam.distributions import Categorical, LogUniform, UniformContinuous

_BASE_MODULE_PATH = __name__


def get_base_config() -> BaseExperimentConfig:
    """Returns a GENERIC base config for SAC with a TMLP network."""
    algorithm_component = SACConfig(
        network=SACTMLPActorCriticConfig(), hyperparam=SACHyperparams()
    )

    config = BaseExperimentConfig(
        algorithm=algorithm_component,
        env=BaseEnvironmentConfig(env_name="placeholder_env"),
        training=BaseTrainingConfig(total_timesteps=int(1e6)),
        logger=LoggerConfig(base_exp_path="results_base/sac_tmlp_base"),
        config_name="sac_tmlp_base",
        _base_module_path_=_BASE_MODULE_PATH,
    )
    return config


def get_base_hyperparam_distributions() -> dict[str, Any]:
    arch_choices = [
        (32, 5),
        (64, 6),
        (128, 7),
    ]
    return {
        "__JOINT_SAMPLING__": {
            "algorithm.network.actor_network.pre_torso.arch_choice": {
                "targets": [
                    "algorithm.network.actor_network.pre_torso.hidden_dim",
                    "algorithm.network.actor_network.pre_torso.num_nodes_to_decompose",
                ],
                "choices": arch_choices,
            },
            "algorithm.network.critic_network.pre_torso.arch_choice": {
                "targets": [
                    "algorithm.network.critic_network.pre_torso.hidden_dim",
                    "algorithm.network.critic_network.pre_torso.num_nodes_to_decompose",
                ],
                "choices": arch_choices,
            },
        },
        # SACVectorizedHyperparams
        "algorithm.hyperparam.actor_lr": LogUniform(domain=(1e-5, 1e-3)),
        "algorithm.hyperparam.q_lr": LogUniform(domain=(1e-5, 1e-3)),
        "algorithm.hyperparam.alpha_lr": LogUniform(domain=(1e-5, 1e-3)),
        "algorithm.hyperparam.gamma": UniformContinuous(domain=(0.9, 0.999)),
        "algorithm.hyperparam.tau": UniformContinuous(domain=(0.001, 0.1)),
        "algorithm.hyperparam.max_grad_norm": UniformContinuous(domain=(0.1, 1.0)),
        "algorithm.hyperparam.target_entropy_scale": UniformContinuous(domain=(0.5, 1.0)),
        "algorithm.hyperparam.total_num_envs": Categorical(values=[1024, 2048, 4096]),
        "algorithm.hyperparam.rollout_length": Categorical(values=[1, 2, 4, 8]),
        "algorithm.hyperparam.warmup_rollout_length": Categorical(values=[8, 16, 32]),
        "algorithm.hyperparam.epochs": Categorical(values=[1, 2, 4]),
        "algorithm.hyperparam.total_buffer_size": Categorical(values=[int(3e4), int(1e5)]),
        "algorithm.hyperparam.total_batch_size": Categorical(values=[128, 256, 512]),
        "algorithm.hyperparam.normalize_observations": Categorical(values=[True]),
        # SACNonVecHyperparams
        "algorithm.hyperparam.autotune": Categorical(values=[True]),
        # TMLP Independent Params
        "algorithm.network.actor_network.pre_torso.bond_dim": Categorical(values=[4, 8, 16]),
        "algorithm.network.actor_network.pre_torso.activation": Categorical(
            values=["silu", "relu", "tanh"]
        ),
        "algorithm.network.actor_network.pre_torso.num_mpo_layers": Categorical(values=[1, 2, 3]),
        "algorithm.network.critic_network.pre_torso.bond_dim": Categorical(values=[4, 8, 16]),
        "algorithm.network.critic_network.pre_torso.activation": Categorical(
            values=["silu", "relu", "tanh"]
        ),
        "algorithm.network.critic_network.pre_torso.num_mpo_layers": Categorical(values=[1, 2, 3]),
        # Proxy Samplers
        "algorithm.network.actor_network.pre_torso.arch_choice": Categorical(
            values=list(range(len(arch_choices)))
        ),
        "algorithm.network.critic_network.pre_torso.arch_choice": Categorical(
            values=list(range(len(arch_choices)))
        ),
    }


def get_base_hyperparam_distributions_test_quick() -> dict[str, Any]:
    arch_choices = [[32, 5]]
    return {
        "__JOINT_SAMPLING__": {
            "algorithm.network.actor_network.pre_torso.arch_choice": {
                "targets": [
                    "algorithm.network.actor_network.pre_torso.hidden_dim",
                    "algorithm.network.actor_network.pre_torso.num_nodes_to_decompose",
                ],
                "choices": arch_choices,
            },
            "algorithm.network.critic_network.pre_torso.arch_choice": {
                "targets": [
                    "algorithm.network.critic_network.pre_torso.hidden_dim",
                    "algorithm.network.critic_network.pre_torso.num_nodes_to_decompose",
                ],
                "choices": arch_choices,
            },
        },
        # SACVectorizedHyperparams (quick test)
        "algorithm.hyperparam.actor_lr": LogUniform(domain=(1e-4, 1e-3)),
        "algorithm.hyperparam.q_lr": LogUniform(domain=(1e-4, 1e-3)),
        "algorithm.hyperparam.alpha_lr": LogUniform(domain=(1e-4, 1e-3)),
        "algorithm.hyperparam.gamma": UniformContinuous(domain=(0.98, 0.99)),
        "algorithm.hyperparam.tau": UniformContinuous(domain=(0.005, 0.01)),
        "algorithm.hyperparam.max_grad_norm": UniformContinuous(domain=(0.1, 1.0)),
        "algorithm.hyperparam.target_entropy_scale": UniformContinuous(domain=(0.8, 0.9)),
        "algorithm.hyperparam.init_alpha": LogUniform(domain=(0.5, 1.5)),
        "algorithm.hyperparam.total_num_envs": Categorical(values=[2, 4]),
        "algorithm.hyperparam.rollout_length": Categorical(values=[1, 2]),
        "algorithm.hyperparam.warmup_rollout_length": Categorical(values=[2, 4]),
        "algorithm.hyperparam.epochs": Categorical(values=[1, 2]),
        "algorithm.hyperparam.total_buffer_size": Categorical(values=[100, 200]),
        "algorithm.hyperparam.total_batch_size": Categorical(values=[1, 2]),
        "algorithm.hyperparam.normalize_observations": Categorical(values=[True]),
        # SACNonVecHyperparams (only autotune)
        "algorithm.hyperparam.autotune": Categorical(values=[True]),
        # TMLP independent params (quick test)
        "algorithm.network.actor_network.pre_torso.bond_dim": Categorical(values=[4]),
        "algorithm.network.actor_network.pre_torso.activation": Categorical(values=["relu"]),
        "algorithm.network.actor_network.pre_torso.num_mpo_layers": Categorical(values=[1]),
        "algorithm.network.critic_network.pre_torso.bond_dim": Categorical(values=[4]),
        "algorithm.network.critic_network.pre_torso.activation": Categorical(values=["relu"]),
        "algorithm.network.critic_network.pre_torso.num_mpo_layers": Categorical(values=[1]),
        # Proxy Samplers
        "algorithm.network.actor_network.pre_torso.arch_choice": Categorical(
            values=list(range(len(arch_choices)))
        ),
        "algorithm.network.critic_network.pre_torso.arch_choice": Categorical(
            values=list(range(len(arch_choices)))
        ),
    }
