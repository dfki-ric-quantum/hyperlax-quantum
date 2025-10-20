from typing import Any

from hyperlax.algo.ppo.config import PPOConfig
from hyperlax.algo.ppo.hyperparam import PPOHyperparams
from hyperlax.configs.env.base_env import BaseEnvironmentConfig
from hyperlax.configs.logger.base_logger import LoggerConfig
from hyperlax.configs.main_base import BaseExperimentConfig
from hyperlax.configs.network.actorcritic_drpqc import DRPQCActorCriticConfig
from hyperlax.configs.training.base_training import BaseTrainingConfig
from hyperlax.hyperparam.distributions import Categorical, LogUniform, UniformContinuous

_BASE_MODULE_PATH = __name__


def get_base_config() -> BaseExperimentConfig:
    algorithm_component = PPOConfig(
        network=DRPQCActorCriticConfig(),
        hyperparam=PPOHyperparams(),
    )

    config = BaseExperimentConfig(
        algorithm=algorithm_component,
        env=BaseEnvironmentConfig(),
        training=BaseTrainingConfig(),
        logger=LoggerConfig(),
        config_name="ppo_drpqc_base",
        _base_module_path_=_BASE_MODULE_PATH,
    )
    return config


def get_base_hyperparam_distributions() -> dict[str, Any]:
    return {
        "algorithm.hyperparam.actor_lr": LogUniform(domain=(1e-5, 1e-3)),
        "algorithm.hyperparam.critic_lr": LogUniform(domain=(1e-5, 1e-3)),
        "algorithm.hyperparam.gamma": UniformContinuous(domain=(0.9, 0.99)),
        "algorithm.hyperparam.gae_lambda": UniformContinuous(domain=(0.9, 0.99)),
        "algorithm.hyperparam.clip_eps": UniformContinuous(domain=(0.01, 0.5)),
        "algorithm.hyperparam.ent_coef": LogUniform(domain=(1e-5, 1e-1)),
        "algorithm.hyperparam.vf_coef": UniformContinuous(domain=(0.5, 1.0)),
        "algorithm.hyperparam.max_grad_norm": UniformContinuous(domain=(0.1, 1.0)),
        "algorithm.hyperparam.rollout_length": Categorical(values=[1, 2, 4, 8]),
        "algorithm.hyperparam.epochs": Categorical(values=[1, 2, 4]),
        "algorithm.hyperparam.num_minibatches": Categorical(values=[128, 256, 512]),
        "algorithm.hyperparam.total_num_envs": Categorical(values=[1024, 2048, 4096]),
        "algorithm.hyperparam.standardize_advantages": Categorical(values=[True, False]),
        "algorithm.hyperparam.decay_learning_rates": Categorical(values=[True, False]),
        "algorithm.hyperparam.normalize_observations": Categorical(values=[True]),
        "algorithm.network.actor_network.pre_torso.n_layers": Categorical(values=[5, 10, 15]),
        "algorithm.network.actor_network.pre_torso.n_vstack": Categorical(values=[1, 2, 3]),
        "algorithm.network.critic_network.pre_torso.n_layers": Categorical(values=[5, 10, 15]),
        "algorithm.network.critic_network.pre_torso.n_vstack": Categorical(values=[1, 2, 3]),
    }

