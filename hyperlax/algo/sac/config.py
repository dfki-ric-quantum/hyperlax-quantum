from dataclasses import dataclass, field

from hyperlax.algo.sac.hyperparam import SACHyperparams
from hyperlax.configs.network.actorcritic_sac_drpqc import SACDRPQCActorCriticConfig
from hyperlax.configs.network.actorcritic_sac_mlp import SACMLPActorCriticConfig
from hyperlax.configs.network.actorcritic_sac_tmlp import SACTMLPActorCriticConfig


@dataclass
class SACConfig:
    """Component config for the SAC algorithm."""

    _target_: str = "hyperlax.algo.sac.main_sac.main"
    network: SACMLPActorCriticConfig | SACDRPQCActorCriticConfig | SACTMLPActorCriticConfig = (
        field(default_factory=SACMLPActorCriticConfig)
    )
    hyperparam: SACHyperparams = field(default_factory=SACHyperparams)
