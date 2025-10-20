from typing import Any

from hyperlax.algo.dqn.config import DQNConfig
from hyperlax.algo.dqn.hyperparam import DQNHyperparams
from hyperlax.configs.env.base_env import BaseEnvironmentConfig
from hyperlax.configs.logger.base_logger import LoggerConfig
from hyperlax.configs.main_base import BaseExperimentConfig
from hyperlax.configs.network.critic_mlp import MLPQNetworkConfig
from hyperlax.configs.training.base_training import BaseTrainingConfig
from hyperlax.hyperparam.distributions import Categorical, LogUniform, UniformContinuous

_BASE_MODULE_PATH = __name__


def get_base_config() -> BaseExperimentConfig:
    algorithm_component = DQNConfig(
        network=MLPQNetworkConfig(),
        hyperparam=DQNHyperparams(),
    )

    config = BaseExperimentConfig(
        algorithm=algorithm_component,
        env=BaseEnvironmentConfig(env_name="placeholder_env"),
        training=BaseTrainingConfig(
            total_timesteps=int(1e6),
            num_eval_episodes=128,
            num_evaluation=50,
            num_agents_slash_seeds=8,
        ),
        logger=LoggerConfig(base_exp_path="results_base/dqn_mlp_base"),
        config_name="dqn_mlp_base",
        experiment_tags="dqn,mlp,S1024",
        _base_module_path_=_BASE_MODULE_PATH,
    )
    return config


def get_base_hyperparam_distributions() -> dict[str, Any]:
    return {
        "algorithm.hyperparam.critic_lr": LogUniform(domain=(1e-5, 1e-3)),
        "algorithm.hyperparam.tau": UniformContinuous(domain=(0.001, 0.1)),
        "algorithm.hyperparam.gamma": UniformContinuous(domain=(0.9, 0.99)),
        "algorithm.hyperparam.max_grad_norm": UniformContinuous(domain=(0.1, 1.0)),
        "algorithm.hyperparam.training_epsilon": UniformContinuous(domain=(0.01, 0.2)),
        "algorithm.hyperparam.huber_loss_parameter": UniformContinuous(domain=(0.0, 1.0)),
        "algorithm.hyperparam.warmup_rollout_length": Categorical(values=[16]),
        "algorithm.hyperparam.total_num_envs": Categorical(values=[128]),
        "algorithm.hyperparam.rollout_length": Categorical(values=[4]),
        "algorithm.hyperparam.epochs": Categorical(values=[1]),
        "algorithm.hyperparam.total_buffer_size": Categorical(values=[int(1e5)]),
        "algorithm.hyperparam.total_batch_size": Categorical(values=[64]),
        "algorithm.hyperparam.normalize_observations": Categorical(values=[True]),
        "algorithm.hyperparam.decay_learning_rates": Categorical(values=[False]),
        "algorithm.hyperparam.use_double_q": Categorical(values=[True]),
        "algorithm.network.critic_network.pre_torso.layer_sizes": Categorical(
            values=[[128, 128]]
        ),
        "algorithm.network.critic_network.pre_torso.activation": Categorical(
            values=["relu"]
        ),
        "algorithm.network.critic_network.pre_torso.use_layer_norm": Categorical(
            values=[False]
        ),
    }


def get_base_hyperparam_distributions_test_quick() -> dict[str, Any]:
    return {
        "algorithm.hyperparam.critic_lr": LogUniform(domain=(1e-5, 1e-4)),
        "algorithm.hyperparam.tau": UniformContinuous(domain=(0.001, 0.1)),
        "algorithm.hyperparam.gamma": UniformContinuous(domain=(0.9, 0.9999)),
        "algorithm.hyperparam.max_grad_norm": UniformContinuous(domain=(0.4, 0.6)),
        "algorithm.hyperparam.training_epsilon": UniformContinuous(domain=(0.01, 0.2)),
        "algorithm.hyperparam.huber_loss_parameter": UniformContinuous(
            domain=(0.0, 0.1)
        ),  # 0.0 means L2
        "algorithm.hyperparam.warmup_rollout_length": Categorical(values=[2]),
        "algorithm.hyperparam.total_num_envs": Categorical(values=[2]),
        "algorithm.hyperparam.rollout_length": Categorical(values=[1]),
        "algorithm.hyperparam.epochs": Categorical(values=[1]),
        "algorithm.hyperparam.total_buffer_size": Categorical(values=[100]),
        "algorithm.hyperparam.total_batch_size": Categorical(values=[1]),
        "algorithm.hyperparam.normalize_observations": Categorical(values=[True]),
        "algorithm.hyperparam.decay_learning_rates": Categorical(values=[True, False]),
        "algorithm.hyperparam.use_double_q": Categorical(values=[True]),
        "algorithm.network.critic_network.pre_torso.layer_sizes": Categorical(values=[[8]]),
        "algorithm.network.critic_network.pre_torso.activation": Categorical(values=["relu"]),
        "algorithm.network.critic_network.pre_torso.use_layer_norm": Categorical(values=[False]),
    }
