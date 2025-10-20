from typing import Any

from hyperlax.algo.sac.config import SACConfig
from hyperlax.algo.sac.hyperparam import SACHyperparams
from hyperlax.configs.env.base_env import BaseEnvironmentConfig
from hyperlax.configs.logger.base_logger import LoggerConfig
from hyperlax.configs.main_base import BaseExperimentConfig
from hyperlax.configs.network.actorcritic_sac_mlp import SACMLPActorCriticConfig
from hyperlax.configs.training.base_training import BaseTrainingConfig
from hyperlax.hyperparam.distributions import Categorical, LogUniform, UniformContinuous

_BASE_MODULE_PATH = __name__


def get_base_config() -> BaseExperimentConfig:
    algorithm_component = SACConfig(
        network=SACMLPActorCriticConfig(),
        hyperparam=SACHyperparams())

    config = BaseExperimentConfig(
        algorithm=algorithm_component,
        env=BaseEnvironmentConfig(),
        training=BaseTrainingConfig(),
        logger=LoggerConfig(),
        config_name="sac_mlp_base",
        _base_module_path_=_BASE_MODULE_PATH,
    )
    return config


def get_base_hyperparam_distributions() -> dict[str, Any]:
    return {
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
        "algorithm.hyperparam.autotune": Categorical(values=[True]),
        "algorithm.network.actor_network.pre_torso.layer_sizes": Categorical(
            values=[[32, 32], [64, 64], [128, 128]]
        ),
        "algorithm.network.actor_network.pre_torso.activation": Categorical(
            values=["tanh", "relu", "silu"]
        ),
        "algorithm.network.actor_network.pre_torso.use_layer_norm": Categorical(
            values=[True, False]
        ),
        "algorithm.network.critic_network.pre_torso.layer_sizes": Categorical(
            values=[[32, 32], [64, 64], [128, 128]]
        ),
        "algorithm.network.critic_network.pre_torso.activation": Categorical(
            values=["tanh", "relu", "silu"]
        ),
        "algorithm.network.critic_network.pre_torso.use_layer_norm": Categorical(
            values=[True, False]
        ),
    }
