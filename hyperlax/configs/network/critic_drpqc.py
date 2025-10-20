from dataclasses import dataclass, field

from hyperlax.configs.network.actorcritic_drpqc import (
    DataReuploadingParametrizedQuantumCircuit,
)


@dataclass(frozen=True)
class ActionHeadConfig:
    _target_: str = "hyperlax.network.heads.DiscreteQNetworkHead"


@dataclass(frozen=True)
class DRPQCCriticConfig:
    pre_torso: DataReuploadingParametrizedQuantumCircuit = field(
        default_factory=DataReuploadingParametrizedQuantumCircuit
    )
    critic_head: ActionHeadConfig = field(default_factory=ActionHeadConfig)


@dataclass(frozen=True)
class DRPQCQNetworkConfig:
    critic_network: DRPQCCriticConfig = field(default_factory=DRPQCCriticConfig)
