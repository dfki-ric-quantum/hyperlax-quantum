from dataclasses import dataclass, field

from hyperlax.algo.ppo.hyperparam import PPOHyperparams
from hyperlax.configs.network.actorcritic_drpqc import DRPQCActorCriticConfig
from hyperlax.configs.network.actorcritic_mlp import MLPActorCriticConfig
from hyperlax.configs.network.actorcritic_tmlp import TMLPMPODActorCriticConfig
from hyperlax.configs.network.actorcritic_vec_mlp import (
    MLPActorCriticConfig as VecMLPActorCriticConfig,
)


@dataclass
class PPOConfig:
    """Component config for the PPO algorithm."""

    _target_: str = "hyperlax.algo.ppo.main_ppo.main"
    network: (
        MLPActorCriticConfig
        | TMLPMPODActorCriticConfig
        | DRPQCActorCriticConfig
        | VecMLPActorCriticConfig
    ) = field(default_factory=MLPActorCriticConfig)
    hyperparam: PPOHyperparams = field(default_factory=PPOHyperparams)
