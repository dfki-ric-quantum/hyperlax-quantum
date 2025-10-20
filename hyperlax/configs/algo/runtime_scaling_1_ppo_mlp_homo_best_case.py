from typing import Any

from hyperlax.algo.ppo.config import PPOConfig
from hyperlax.algo.ppo.hyperparam import PPOHyperparams
from hyperlax.configs.env.base_env import BaseEnvironmentConfig
from hyperlax.configs.logger.base_logger import LoggerConfig
from hyperlax.configs.main_base import BaseExperimentConfig
from hyperlax.configs.network.actorcritic_mlp import MLPActorCriticConfig
from hyperlax.configs.training.base_training import BaseTrainingConfig
from hyperlax.hyperparam.distributions import Categorical, LogUniform, UniformContinuous

_BASE_MODULE_PATH = __name__


def get_base_config() -> BaseExperimentConfig:
    """
    Returns a GENERIC base config for PPO with an MLP.
    The action head in the network config is a placeholder and will be
    dynamically replaced by the runner based on the environment's action space.
    """
    algorithm_component = PPOConfig(network=MLPActorCriticConfig(), hyperparam=PPOHyperparams())

    config = BaseExperimentConfig(
        algorithm=algorithm_component,
        env=BaseEnvironmentConfig(env_name="placeholder_env"),
        training=BaseTrainingConfig(total_timesteps=int(1e6)),
        logger=LoggerConfig(base_exp_path="results_base/ppo_mlp_base"),
        config_name="ppo_mlp_base",
        _base_module_path_=_BASE_MODULE_PATH,
    )
    return config


# NOTE Import a base MLP config (w/ continuous as default).
# We will dynamically set the head based on env action space.


def get_base_hyperparam_distributions() -> dict[str, Any]:
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
        # non-vec hyperparams
        "algorithm.network.actor_network.pre_torso.layer_sizes": Categorical(values=[[128, 128]]),
        "algorithm.network.actor_network.pre_torso.activation": Categorical(values=["relu"]),
        "algorithm.network.actor_network.pre_torso.use_layer_norm": Categorical(values=[False]),
        "algorithm.network.critic_network.pre_torso.layer_sizes": Categorical(values=[[128, 128]]),
        "algorithm.network.critic_network.pre_torso.activation": Categorical(values=["relu"]),
        "algorithm.network.critic_network.pre_torso.use_layer_norm": Categorical(values=[False]),
    }


def get_base_hyperparam_distributions_test_quick() -> dict[str, Any]:
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
        # non-vec hyperparams
        "algorithm.network.actor_network.pre_torso.layer_sizes": Categorical(values=[[8]]),
        "algorithm.network.actor_network.pre_torso.activation": Categorical(values=["relu"]),
        "algorithm.network.actor_network.pre_torso.use_layer_norm": Categorical(values=[False]),
        "algorithm.network.critic_network.pre_torso.layer_sizes": Categorical(values=[[8]]),
        "algorithm.network.critic_network.pre_torso.activation": Categorical(values=["relu"]),
        "algorithm.network.critic_network.pre_torso.use_layer_norm": Categorical(values=[False]),
    }
