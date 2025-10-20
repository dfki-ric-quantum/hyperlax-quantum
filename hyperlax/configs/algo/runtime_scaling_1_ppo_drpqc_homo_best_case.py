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
    """
    Returns a GENERIC base config for PPO with a PQC network.
    The action head is configured for continuous action spaces by default and
    will be dynamically patched by the launcher for discrete environments.
    """
    algorithm_component = PPOConfig(
        network=DRPQCActorCriticConfig(),
        hyperparam=PPOHyperparams(),
    )

    config = BaseExperimentConfig(
        algorithm=algorithm_component,
        env=BaseEnvironmentConfig(env_name="placeholder_env"),
        training=BaseTrainingConfig(total_timesteps=int(1e6)),
        logger=LoggerConfig(base_exp_path="results_base/ppo_pqc_base"),
        config_name="ppo_pqc_base",
        _base_module_path_=_BASE_MODULE_PATH,
    )
    return config


def get_base_hyperparam_distributions() -> dict[str, Any]:
    """Returns the base hyperparameter distributions for PPO with a PQC network."""
    return {
        # vec floats aka homogeneous shape
        "algorithm.hyperparam.actor_lr": LogUniform(domain=(1e-5, 1e-2)),
        "algorithm.hyperparam.critic_lr": LogUniform(domain=(1e-5, 1e-2)),
        "algorithm.hyperparam.gamma": UniformContinuous(domain=(0.9, 0.99)),
        "algorithm.hyperparam.gae_lambda": UniformContinuous(domain=(0.9, 0.99)),
        "algorithm.hyperparam.clip_eps": UniformContinuous(domain=(0.01, 0.5)),
        "algorithm.hyperparam.ent_coef": LogUniform(domain=(1e-5, 1e-2)),
        "algorithm.hyperparam.vf_coef": UniformContinuous(domain=(0.5, 1.0)),
        "algorithm.hyperparam.max_grad_norm": UniformContinuous(domain=(0.1, 1.0)),
        # below hyperparams are heteregenuous shape
        # vec integers
        "algorithm.hyperparam.rollout_length": Categorical(values=[2]),
        "algorithm.hyperparam.epochs": Categorical(values=[1]),
        # "epochs": Categorical(values=[1, 5, 10]),
        "algorithm.hyperparam.num_minibatches": Categorical(values=[256]),
        "algorithm.hyperparam.total_num_envs": Categorical(values=[1024]),
        # vec booleans
        "algorithm.hyperparam.standardize_advantages": Categorical(values=[True]),
        "algorithm.hyperparam.decay_learning_rates": Categorical(values=[True]),
        "algorithm.hyperparam.normalize_observations": Categorical(values=[True]),

        "algorithm.network.actor_network.pre_torso.n_layers": Categorical(values=[10]),
        "algorithm.network.actor_network.pre_torso.n_vstack": Categorical(values=[4]),
        "algorithm.network.critic_network.pre_torso.n_layers": Categorical(values=[10]),
        "algorithm.network.critic_network.pre_torso.n_vstack": Categorical(values=[4]),
    }


def get_base_hyperparam_distributions_test_quick() -> dict[str, Any]:
    """Returns quick test hyperparameter distributions for PPO with a PQC network."""
    return {
        # vec floats
        "algorithm.hyperparam.actor_lr": LogUniform(domain=(1e-5, 1e-3)),
        "algorithm.hyperparam.critic_lr": LogUniform(domain=(1e-5, 1e-3)),
        "algorithm.hyperparam.gamma": UniformContinuous(domain=(0.9, 0.99)),
        "algorithm.hyperparam.gae_lambda": UniformContinuous(domain=(0.9, 0.99)),
        "algorithm.hyperparam.clip_eps": UniformContinuous(domain=(0.01, 0.5)),
        "algorithm.hyperparam.ent_coef": LogUniform(domain=(1e-5, 1e-1)),
        "algorithm.hyperparam.vf_coef": UniformContinuous(domain=(0.5, 1.0)),
        "algorithm.hyperparam.max_grad_norm": UniformContinuous(domain=(0.1, 1.0)),
        # vec integers
        "algorithm.hyperparam.rollout_length": Categorical(values=[2]),
        "algorithm.hyperparam.epochs": Categorical(values=[2]),
        "algorithm.hyperparam.num_minibatches": Categorical(values=[2]),
        "algorithm.hyperparam.total_num_envs": Categorical(values=[4]),
        # vec booleans
        "algorithm.hyperparam.standardize_advantages": Categorical(values=[True]),
        "algorithm.hyperparam.decay_learning_rates": Categorical(values=[True]),
        "algorithm.hyperparam.normalize_observations": Categorical(values=[True]),

        "algorithm.network.actor_network.pre_torso.n_layers": Categorical(values=[1]),
        "algorithm.network.actor_network.pre_torso.n_vstack": Categorical(values=[1]),
        "algorithm.network.critic_network.pre_torso.n_layers": Categorical(values=[1]),
        "algorithm.network.critic_network.pre_torso.n_vstack": Categorical(values=[1]),
    }
