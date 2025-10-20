from dataclasses import dataclass, field

from hyperlax.configs.network.actorcritic_tmlp import MPODecomposedMLPTorso


@dataclass(frozen=True)
class ActionHeadConfig:
    _target_: str = "hyperlax.network.heads.DiscreteQNetworkHead"


@dataclass(frozen=True)
class TMLPCriticConfig:
    pre_torso: MPODecomposedMLPTorso = field(default_factory=MPODecomposedMLPTorso)
    critic_head: ActionHeadConfig = field(default_factory=ActionHeadConfig)


@dataclass(frozen=True)
class TMLPQNetworkConfig:
    critic_network: TMLPCriticConfig = field(default_factory=TMLPCriticConfig)
