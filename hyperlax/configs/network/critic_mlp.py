from dataclasses import dataclass, field

from hyperlax.configs.network.actorcritic_mlp import MLPTorso


@dataclass(frozen=True)
class ActionHeadConfig:
    _target_: str = "hyperlax.network.heads.DiscreteQNetworkHead"


@dataclass(frozen=True)
class MLPCriticConfig:
    pre_torso: MLPTorso = field(default_factory=MLPTorso)
    critic_head: ActionHeadConfig = field(default_factory=ActionHeadConfig)


@dataclass(frozen=True)
class MLPQNetworkConfig:
    critic_network: MLPCriticConfig = field(default_factory=MLPCriticConfig)
