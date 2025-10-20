from dataclasses import dataclass, field

from hyperlax.configs.network.actorcritic_drpqc import (
    DataReuploadingParametrizedQuantumCircuit,
)


@dataclass(frozen=True)
class ActorActionHead:
    _target_: str = "hyperlax.network.heads.NormalAffineTanhDistributionHead"
    minimum: float = -1.0
    maximum: float = 1.0


@dataclass(frozen=True)
class CriticInputLayer:
    _target_: str = "hyperlax.network.inputs.ObservationActionInput"


@dataclass(frozen=True)
class CriticValueHead:
    _target_: str = "hyperlax.network.heads.ScalarCriticHead"


@dataclass(frozen=True)
class SACDRPQCActorNetworkConfig:
    pre_torso: DataReuploadingParametrizedQuantumCircuit = field(
        default_factory=DataReuploadingParametrizedQuantumCircuit
    )
    action_head: ActorActionHead = field(default_factory=ActorActionHead)


@dataclass(frozen=True)
class SACDRPQCCriticNetworkConfig:
    input_layer: CriticInputLayer = field(default_factory=CriticInputLayer)
    pre_torso: DataReuploadingParametrizedQuantumCircuit = field(
        default_factory=DataReuploadingParametrizedQuantumCircuit
    )
    critic_head: CriticValueHead = field(default_factory=CriticValueHead)


@dataclass(frozen=True)
class SACDRPQCActorCriticConfig:
    actor_network: SACDRPQCActorNetworkConfig = field(default_factory=SACDRPQCActorNetworkConfig)
    critic_network: SACDRPQCCriticNetworkConfig = field(
        default_factory=SACDRPQCCriticNetworkConfig
    )
