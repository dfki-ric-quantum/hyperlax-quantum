from dataclasses import dataclass, field

from hyperlax.configs.network.actorcritic_tmlp import MPODecomposedMLPTorso


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
class SACTMLPActorNetworkConfig:
    pre_torso: MPODecomposedMLPTorso = field(default_factory=MPODecomposedMLPTorso)
    action_head: ActorActionHead = field(default_factory=ActorActionHead)


@dataclass(frozen=True)
class SACTMLPCriticNetworkConfig:
    input_layer: CriticInputLayer = field(default_factory=CriticInputLayer)
    pre_torso: MPODecomposedMLPTorso = field(default_factory=MPODecomposedMLPTorso)
    critic_head: CriticValueHead = field(default_factory=CriticValueHead)


@dataclass(frozen=True)
class SACTMLPActorCriticConfig:
    actor_network: SACTMLPActorNetworkConfig = field(default_factory=SACTMLPActorNetworkConfig)
    critic_network: SACTMLPCriticNetworkConfig = field(default_factory=SACTMLPCriticNetworkConfig)
