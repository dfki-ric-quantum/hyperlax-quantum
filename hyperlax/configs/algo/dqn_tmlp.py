from typing import Any

from hyperlax.algo.dqn.config import DQNConfig
from hyperlax.algo.dqn.hyperparam import DQNHyperparams
from hyperlax.configs.env.base_env import BaseEnvironmentConfig
from hyperlax.configs.logger.base_logger import LoggerConfig
from hyperlax.configs.main_base import BaseExperimentConfig
from hyperlax.configs.network.critic_tmlp import TMLPQNetworkConfig
from hyperlax.configs.training.base_training import BaseTrainingConfig
from hyperlax.hyperparam.distributions import Categorical, LogUniform, UniformContinuous

_BASE_MODULE_PATH = __name__


def get_base_config() -> BaseExperimentConfig:
    """Returns the base config for DQN with a TMLP network."""
    algorithm_component = DQNConfig(
        network=TMLPQNetworkConfig(),
        hyperparam=DQNHyperparams(),
    )

    config = BaseExperimentConfig(
        algorithm=algorithm_component,
        env=BaseEnvironmentConfig(env_name="placeholder_env"),
        training=BaseTrainingConfig(total_timesteps=int(1e6)),
        logger=LoggerConfig(base_exp_path="results_base/dqn_tmlp_base"),
        config_name="dqn_tmlp_base",
        experiment_mode="batched",
        experiment_tags="dqn_tmlp_base",
        _base_module_path_=_BASE_MODULE_PATH,
    )
    return config


def get_base_hyperparam_distributions() -> dict[str, Any]:
    """Returns the base hyperparameter distributions for DQN with a TMLP network."""
    # Define the joint choices for critic network architecture.
    # Each tuple contains values for (hidden_dim, num_nodes_to_decompose).
    critic_arch_choices = [
        (32, 5),  # hidden_dim=32, num_nodes_to_decompose=5
        (64, 6),  # hidden_dim=64, num_nodes_to_decompose=6
        (128, 7),  # hidden_dim=128, num_nodes_to_decompose=7
    ]

    return {
        # Special key to define joint sampling rules
        "__JOINT_SAMPLING__": {
            "algorithm.network.critic_network.pre_torso.arch_choice": {
                "targets": [
                    "algorithm.network.critic_network.pre_torso.hidden_dim",
                    "algorithm.network.critic_network.pre_torso.num_nodes_to_decompose",
                ],
                "choices": critic_arch_choices,
            }
        },
        "algorithm.hyperparam.critic_lr": LogUniform(domain=(1e-5, 1e-3)),
        "algorithm.hyperparam.tau": UniformContinuous(domain=(0.001, 0.1)),
        "algorithm.hyperparam.gamma": UniformContinuous(domain=(0.9, 0.99)),
        "algorithm.hyperparam.max_grad_norm": UniformContinuous(domain=(0.1, 1.0)),
        "algorithm.hyperparam.training_epsilon": UniformContinuous(domain=(0.01, 0.2)),
        "algorithm.hyperparam.huber_loss_parameter": UniformContinuous(domain=(0.0, 1.0)),
        "algorithm.hyperparam.warmup_rollout_length": Categorical(values=[8, 16, 32]),
        "algorithm.hyperparam.total_num_envs": Categorical(values=[1024, 2048, 4096]),
        "algorithm.hyperparam.rollout_length": Categorical(values=[1, 2, 4, 8]),
        "algorithm.hyperparam.epochs": Categorical(values=[1, 2, 4]),
        "algorithm.hyperparam.total_buffer_size": Categorical(values=[int(3e4), int(1e5)]),
        "algorithm.hyperparam.total_batch_size": Categorical(values=[128, 256, 512]),
        "algorithm.hyperparam.normalize_observations": Categorical(values=[True]),
        "algorithm.hyperparam.decay_learning_rates": Categorical(values=[True, False]),
        "algorithm.hyperparam.use_double_q": Categorical(values=[True]),
        # The proxy parameter that will actually be sampled
        "algorithm.network.critic_network.pre_torso.arch_choice": Categorical(
            values=list(range(len(critic_arch_choices)))
        ),
        # Independent parameters are still defined normally
        "algorithm.network.critic_network.pre_torso.bond_dim": Categorical(values=[4, 8, 16]),
        "algorithm.network.critic_network.pre_torso.activation": Categorical(
            values=["silu", "relu", "tanh"]
        ),
        "algorithm.network.critic_network.pre_torso.num_mpo_layers": Categorical(values=[2]),
    }


def get_base_hyperparam_distributions_test_quick() -> dict[str, Any]:
    """Returns quick test hyperparameter distributions for DQN with a TMLP network."""
    return {
        "algorithm.hyperparam.critic_lr": LogUniform(domain=(1e-5, 1e-4)),
        "algorithm.hyperparam.tau": UniformContinuous(domain=(0.001, 0.1)),
        "algorithm.hyperparam.gamma": UniformContinuous(domain=(0.9, 0.9999)),
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
        "algorithm.hyperparam.decay_learning_rates": Categorical(values=[True, False]),
        "algorithm.hyperparam.use_double_q": Categorical(values=[True]),
        "algorithm.network.critic_network.pre_torso.hidden_dim": Categorical(values=[8]),
        "algorithm.network.critic_network.pre_torso.num_nodes_to_decompose": Categorical(
            values=[3]
        ),
        "algorithm.network.critic_network.pre_torso.bond_dim": Categorical(values=[4]),
        "algorithm.network.critic_network.pre_torso.activation": Categorical(values=["relu"]),
        "algorithm.network.critic_network.pre_torso.num_mpo_layers": Categorical(values=[1]),
    }
