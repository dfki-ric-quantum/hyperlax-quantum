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
    """Returns a GENERIC base config for SAC with an MLP."""
    algorithm_component = SACConfig(network=SACMLPActorCriticConfig(), hyperparam=SACHyperparams())

    config = BaseExperimentConfig(
        algorithm=algorithm_component,
        env=BaseEnvironmentConfig(env_name="placeholder_env"),
        training=BaseTrainingConfig(
            total_timesteps=int(1e6),
            num_eval_episodes=128,
            num_evaluation=50,
            num_agents_slash_seeds=8,
        ),
        logger=LoggerConfig(base_exp_path="results_base/sac_mlp_base"),
        config_name="sac_mlp_base",
        experiment_tags="sac,mlp,S1024",
        _base_module_path_=_BASE_MODULE_PATH,
    )
    return config


def get_base_hyperparam_distributions() -> dict[str, Any]:
    return {
        # SACVectorizedHyperparams
        "algorithm.hyperparam.actor_lr": LogUniform(domain=(1e-5, 1e-2)),
        "algorithm.hyperparam.q_lr": LogUniform(domain=(1e-5, 1e-2)),
        "algorithm.hyperparam.alpha_lr": LogUniform(domain=(1e-5, 1e-2)),
        "algorithm.hyperparam.gamma": UniformContinuous(domain=(0.9, 0.99)),
        "algorithm.hyperparam.tau": UniformContinuous(domain=(0.001, 0.1)),
        "algorithm.hyperparam.max_grad_norm": UniformContinuous(domain=(0.1, 1.0)),
        "algorithm.hyperparam.target_entropy_scale": UniformContinuous(domain=(0.5, 1.0)),
        # "algorithm.hyperparam.init_alpha": LogUniform(domain=(0.1, 10.0)),
        "algorithm.hyperparam.total_num_envs": Categorical(values=[128]),
        "algorithm.hyperparam.rollout_length": Categorical(values=[4]),
        "algorithm.hyperparam.warmup_rollout_length": Categorical(values=[16]),
        "algorithm.hyperparam.epochs": Categorical(values=[1]),
        "algorithm.hyperparam.total_buffer_size": Categorical(values=[int(1e5)]),
        "algorithm.hyperparam.total_batch_size": Categorical(values=[64]),
        "algorithm.hyperparam.normalize_observations": Categorical(values=[True]),
        # SACNonVecHyperparams (only autotune)
        "algorithm.hyperparam.autotune": Categorical(values=[True]),
        # MLPTorso parameters (non-vectorized in this config)
        "algorithm.network.actor_network.pre_torso.layer_sizes": Categorical(
            values=[[128, 128]]
        ),
        "algorithm.network.actor_network.pre_torso.activation": Categorical(
            values=["relu"]
        ),
        "algorithm.network.actor_network.pre_torso.use_layer_norm": Categorical(
            values=[False]
        ),
        "algorithm.network.critic_network.pre_torso.layer_sizes": Categorical(
            values=[[128, 128]]
        ),
        "algorithm.network.critic_network.pre_torso.activation": Categorical(
            values=["relu"]
        ),
        "algorithm.network.critic_network.pre_torso.use_layer_norm": Categorical(
            values=[ False]
        ),
    }


def get_base_hyperparam_distributions_test_quick() -> dict[str, Any]:
    return {
        # SACVectorizedHyperparams (quick test)
        "algorithm.hyperparam.actor_lr": LogUniform(domain=(1e-4, 1e-3)),
        "algorithm.hyperparam.q_lr": LogUniform(domain=(1e-4, 1e-3)),
        "algorithm.hyperparam.alpha_lr": LogUniform(domain=(1e-4, 1e-3)),
        "algorithm.hyperparam.gamma": UniformContinuous(domain=(0.98, 0.99)),
        "algorithm.hyperparam.tau": UniformContinuous(domain=(0.005, 0.01)),
        "algorithm.hyperparam.max_grad_norm": UniformContinuous(domain=(0.1, 1.0)),
        "algorithm.hyperparam.target_entropy_scale": UniformContinuous(domain=(0.8, 0.9)),
        # "algorithm.hyperparam.init_alpha": LogUniform(domain=(0.5, 1.5)),
        "algorithm.hyperparam.total_num_envs": Categorical(values=[4]),
        "algorithm.hyperparam.rollout_length": Categorical(values=[2]),
        "algorithm.hyperparam.warmup_rollout_length": Categorical(values=[4]),
        "algorithm.hyperparam.epochs": Categorical(values=[2]),
        "algorithm.hyperparam.total_buffer_size": Categorical(values=[200]),
        "algorithm.hyperparam.total_batch_size": Categorical(values=[2]),
        "algorithm.hyperparam.normalize_observations": Categorical(values=[True]),
        # SACNonVecHyperparams (only autotune)
        "algorithm.hyperparam.autotune": Categorical(values=[True]),
        # MLPTorso parameters (non-vectorized in this config)
        "algorithm.network.actor_network.pre_torso.layer_sizes": Categorical(
            values=[[8, 8]]
        ),
        "algorithm.network.actor_network.pre_torso.activation": Categorical(
            values=["relu"]
        ),
        "algorithm.network.actor_network.pre_torso.use_layer_norm": Categorical(
            values=[False]
        ),
        "algorithm.network.critic_network.pre_torso.layer_sizes": Categorical(
            values=[[8, 8]]
        ),
        "algorithm.network.critic_network.pre_torso.activation": Categorical(
            values=["relu"]
        ),
        "algorithm.network.critic_network.pre_torso.use_layer_norm": Categorical(
            values=[False]
        ),
    }
