from dataclasses import dataclass, field

from hyperlax.configs.network.actorcritic_mlp import MLPTorso


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
class SACMLPActorNetworkConfig:
    pre_torso: MLPTorso = field(default_factory=MLPTorso)
    action_head: ActorActionHead = field(default_factory=ActorActionHead)


@dataclass(frozen=True)
class SACMLPCriticNetworkConfig:
    input_layer: CriticInputLayer = field(default_factory=CriticInputLayer)
    pre_torso: MLPTorso = field(default_factory=MLPTorso)
    critic_head: CriticValueHead = field(default_factory=CriticValueHead)


@dataclass(frozen=True)
class SACMLPActorCriticConfig:
    actor_network: SACMLPActorNetworkConfig = field(default_factory=SACMLPActorNetworkConfig)
    # The 'critic_network' here defines the architecture for a SINGLE Q-network.
    # The setup logic will create an ensemble of two of these.
    critic_network: SACMLPCriticNetworkConfig = field(default_factory=SACMLPCriticNetworkConfig)
