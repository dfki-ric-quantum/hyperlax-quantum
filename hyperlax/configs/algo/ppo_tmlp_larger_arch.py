from typing import Any

from hyperlax.algo.ppo.config import PPOConfig
from hyperlax.algo.ppo.hyperparam import PPOHyperparams
from hyperlax.configs.env.base_env import BaseEnvironmentConfig
from hyperlax.configs.logger.base_logger import LoggerConfig
from hyperlax.configs.main_base import BaseExperimentConfig
from hyperlax.configs.network.actorcritic_tmlp import TMLPMPODActorCriticConfig
from hyperlax.configs.training.base_training import BaseTrainingConfig
from hyperlax.hyperparam.distributions import Categorical, LogUniform, UniformContinuous

_BASE_MODULE_PATH = __name__


def get_base_config() -> BaseExperimentConfig:
    """
    Returns a GENERIC base config for PPO with a TMLP network.
    The action head is configured for continuous action spaces by default and
    will be dynamically patched by the launcher for discrete environments.
    """
    algorithm_component = PPOConfig(
        network=TMLPMPODActorCriticConfig(),
        hyperparam=PPOHyperparams(),
    )

    config = BaseExperimentConfig(
        algorithm=algorithm_component,
        env=BaseEnvironmentConfig(env_name="placeholder_env"),
        training=BaseTrainingConfig(total_timesteps=int(5e6)),
        logger=LoggerConfig(base_exp_path="results_base/ppo_tmlp_base"),
        config_name="ppo_tmlp_base",
        _base_module_path_=_BASE_MODULE_PATH,
    )
    return config


def get_base_hyperparam_distributions() -> dict[str, Any]:
    """Returns the base hyperparameter distributions for PPO with a TMLP network."""
    # Define joint choices for both actor and critic network architectures.
    # Each tuple: (hidden_dim, num_nodes_to_decompose)
    arch_choices = [
        (64, 6),
        (128, 7),
        (256, 8),
        (512, 9),

        (64, 3),
        (256, 4),

        (64, 2),
        (512, 3),
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
        # PPO Hyperparams (Vectorized)
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
        # TMLP Independent Params (Non-vectorized)
        "algorithm.network.actor_network.pre_torso.bond_dim": Categorical(values=[4, 8, 16]),
        "algorithm.network.actor_network.pre_torso.activation": Categorical(
            values=["silu", "relu", "tanh"]
        ),
        "algorithm.network.actor_network.pre_torso.num_mpo_layers": Categorical(values=[2, 3]),
        "algorithm.network.critic_network.pre_torso.bond_dim": Categorical(values=[4, 8, 16]),
        "algorithm.network.critic_network.pre_torso.activation": Categorical(
            values=["silu", "relu", "tanh"]
        ),
        "algorithm.network.critic_network.pre_torso.num_mpo_layers": Categorical(values=[2, 3]),
        # Proxy Samplers
        "algorithm.network.actor_network.pre_torso.arch_choice": Categorical(
            values=list(range(len(arch_choices)))
        ),
        "algorithm.network.critic_network.pre_torso.arch_choice": Categorical(
            values=list(range(len(arch_choices)))
        ),
    }


def get_base_hyperparam_distributions_test_quick() -> dict[str, Any]:
    """Returns quick test hyperparameter distributions for PPO with a TMLP network."""
    arch_choices = [[32, 5]]  # Single choice for quick test

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
        # vec floats
        "algorithm.hyperparam.actor_lr": LogUniform(domain=(1e-5, 1e-2)),
        "algorithm.hyperparam.critic_lr": LogUniform(domain=(1e-5, 1e-2)),
        "algorithm.hyperparam.gamma": UniformContinuous(domain=(0.9, 0.99)),
        "algorithm.hyperparam.gae_lambda": UniformContinuous(domain=(0.9, 0.99)),
        "algorithm.hyperparam.clip_eps": UniformContinuous(domain=(0.01, 0.5)),
        "algorithm.hyperparam.ent_coef": LogUniform(domain=(1e-5, 1e-2)),
        "algorithm.hyperparam.vf_coef": UniformContinuous(domain=(0.5, 1.0)),
        "algorithm.hyperparam.max_grad_norm": UniformContinuous(domain=(0.1, 1.0)),
        # vec integers
        "algorithm.hyperparam.rollout_length": Categorical(values=[1, 2]),
        "algorithm.hyperparam.epochs": Categorical(values=[1, 2]),
        "algorithm.hyperparam.num_minibatches": Categorical(values=[1, 2]),
        "algorithm.hyperparam.total_num_envs": Categorical(values=[2, 4]),
        # vec booleans
        "algorithm.hyperparam.standardize_advantages": Categorical(values=[True, False]),
        "algorithm.hyperparam.decay_learning_rates": Categorical(values=[True, False]),
        "algorithm.hyperparam.normalize_observations": Categorical(values=[True]),
        # Independent TMLP params
        "algorithm.network.critic_network.pre_torso.bond_dim": Categorical(values=[4]),
        "algorithm.network.critic_network.pre_torso.activation": Categorical(values=["relu"]),
        "algorithm.network.critic_network.pre_torso.num_mpo_layers": Categorical(values=[1]),
        "algorithm.network.actor_network.pre_torso.bond_dim": Categorical(values=[4]),
        "algorithm.network.actor_network.pre_torso.activation": Categorical(values=["relu"]),
        "algorithm.network.actor_network.pre_torso.num_mpo_layers": Categorical(values=[1]),
        # Proxy Samplers
        "algorithm.network.actor_network.pre_torso.arch_choice": Categorical(
            values=list(range(len(arch_choices)))
        ),
        "algorithm.network.critic_network.pre_torso.arch_choice": Categorical(
            values=list(range(len(arch_choices)))
        ),
    }
