from typing import Any

from hyperlax.algo.dqn.config import DQNConfig
from hyperlax.algo.dqn.hyperparam import DQNHyperparams
from hyperlax.configs.env.base_env import BaseEnvironmentConfig
from hyperlax.configs.logger.base_logger import LoggerConfig
from hyperlax.configs.main_base import BaseExperimentConfig
from hyperlax.configs.network.critic_drpqc import DRPQCQNetworkConfig
from hyperlax.configs.training.base_training import BaseTrainingConfig
from hyperlax.hyperparam.distributions import Categorical, LogUniform, UniformContinuous

_BASE_MODULE_PATH = __name__


def get_base_config() -> BaseExperimentConfig:
    """Returns the base config for DQN with a PQC network, containing default values."""
    algorithm_component = DQNConfig(
        network=DRPQCQNetworkConfig(),
        hyperparam=DQNHyperparams(),
    )

    config = BaseExperimentConfig(
        algorithm=algorithm_component,
        env=BaseEnvironmentConfig(env_name="placeholder_env"),
        training=BaseTrainingConfig(total_timesteps=int(1e6)),
        logger=LoggerConfig(base_exp_path="results_base/dqn_drpqc_base"),
        config_name="dqn_drpqc_base",
        _base_module_path_=_BASE_MODULE_PATH,
    )
    return config


def get_base_hyperparam_distributions() -> dict[str, Any]:
    """Returns the base hyperparameter distributions for DQN with a PQC network."""
    return {
        "algorithm.hyperparam.critic_lr": LogUniform(domain=(3e-5, 5e-4)),   # prior mode ~1e-4
        "algorithm.hyperparam.tau":       LogUniform(domain=(1e-3, 2e-2)),   # prior mode ~0.005

        "algorithm.hyperparam.gamma": Categorical(
            values=[0.97, 0.98, 0.99],
            probabilities=[0.15, 0.25, 0.60]
        ),

        "algorithm.hyperparam.max_grad_norm": Categorical(
            values=[0.3, 0.5, 1.0],
            probabilities=[0.25, 0.45, 0.3]
        ),

        "algorithm.hyperparam.training_epsilon": UniformContinuous(domain=(0.08, 0.15)),

        "algorithm.hyperparam.huber_loss_parameter": Categorical(
            values=[0.0, 0.5, 1.0, 2.0],
            probabilities=[0.25, 0.25, 0.25, 0.25]
        ),

        "algorithm.hyperparam.warmup_rollout_length": Categorical(
            values=[8, 16, 32],
            probabilities=[0.25, 0.50, 0.25]
        ),

        "algorithm.hyperparam.total_num_envs": Categorical(
            values=[512, 1024, 2048, 4096],
            probabilities=[0.15, 0.35, 0.35, 0.15]
        ),

        "algorithm.hyperparam.rollout_length": Categorical(
            values=[4, 8, 16, 32],
            probabilities=[0.30, 0.40, 0.20, 0.10]
        ),

        "algorithm.hyperparam.epochs": Categorical(
            values=[1, 2, 4, 8],
            probabilities=[0.45, 0.35, 0.15, 0.05]
        ),

        "algorithm.hyperparam.total_buffer_size": Categorical(
            values=[int(1e5), int(5e5), int(1e6)],
            probabilities=[0.20, 0.30, 0.50]
        ),

        "algorithm.hyperparam.total_batch_size": Categorical(
            values=[64, 128, 256, 512],
            probabilities=[0.20, 0.40, 0.30, 0.10]
        ),

        "algorithm.hyperparam.normalize_observations": Categorical(values=[True]),

        "algorithm.hyperparam.use_double_q": Categorical(values=[True]),
        "algorithm.hyperparam.decay_learning_rates": Categorical(values=[True, False]),

        "algorithm.network.critic_network.pre_torso.n_layers": Categorical(values=[5, 10, 15]),
        "algorithm.network.critic_network.pre_torso.n_vstack": Categorical(values=[1, 2, 3]),

    }


def get_base_hyperparam_distributions_test_quick() -> dict[str, Any]:
    """Returns quick test hyperparameter distributions for DQN with a PQC net work."""
    return {
        "algorithm.hyperparam.critic_lr": LogUniform(domain=(1e-5, 1e-4)),
        "algorithm.hyperparam.tau": UniformContinuous(domain=(0.001, 0.1)),
        "algorithm.hyperparam.gamma": UniformContinuous(domain=(0.95, 0.9999)),
        "algorithm.hyperparam.max_grad_norm": UniformContinuous(domain=(0.4, 0.6)),
        "algorithm.hyperparam.training_epsilon": UniformContinuous(domain=(0.01, 0.2)),
        "algorithm.hyperparam.huber_loss_parameter": UniformContinuous(domain=(0.0, 0.1)),
        "algorithm.hyperparam.warmup_rollout_length": Categorical(values=[1, 2]),
        "algorithm.hyperparam.total_num_envs": Categorical(values=[2, 4]),
        "algorithm.hyperparam.rollout_length": Categorical(values=[1, 2]),
        "algorithm.hyperparam.epochs": Categorical(values=[1, 2]),
        "algorithm.hyperparam.total_buffer_size": Categorical(values=[100]),
        "algorithm.hyperparam.total_batch_size": Categorical(values=[1, 2]),
        "algorithm.hyperparam.normalize_observations": Categorical(values=[True]),
        "algorithm.hyperparam.use_double_q": Categorical(values=[True]),
        "algorithm.hyperparam.decay_learning_rates": Categorical(values=[True, False]),
        "algorithm.network.critic_network.pre_torso.n_layers": Categorical(values=[2]),
        "algorithm.network.critic_network.pre_torso.n_vstack": Categorical(values=[3]),
    }
