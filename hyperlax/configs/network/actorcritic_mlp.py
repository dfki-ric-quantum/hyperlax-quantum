from dataclasses import dataclass, field

from hyperlax.hyperparam.distributions import Categorical
from hyperlax.hyperparam.tunable import Tunable


@dataclass(frozen=True)
class MLPTorso:
    _target_: str = "hyperlax.network.torso.MLPTorso"
    layer_sizes: Tunable = field(
        default_factory=lambda: Tunable(
            value=[256, 256],
            is_vectorized=False,
            is_fixed=False,
            distribution=Categorical(values=[[64, 64], [128, 128], [256, 256]]),
            expected_type=list[int],
        )
    )
    use_layer_norm: Tunable = field(
        default_factory=lambda: Tunable(
            value=False,
            is_vectorized=False,
            is_fixed=False,
            distribution=Categorical(values=[True, False]),
            expected_type=bool,
        )
    )
    activation: Tunable = field(
        default_factory=lambda: Tunable(
            value="silu",
            is_vectorized=False,
            is_fixed=False,
            distribution=Categorical(values=["silu", "relu", "tanh"]),
            expected_type=str,
        )
    )


@dataclass(frozen=True)
class ActionHead:
    _target_: str = "hyperlax.network.heads.NormalAffineTanhDistributionHead"
    minimum: float = -999.0
    maximum: float = -999.0


@dataclass(frozen=True)
class CriticHead:
    _target_: str = "hyperlax.network.heads.ScalarCriticHead"


@dataclass(frozen=True)
class MLPActorNetworkConfig:
    pre_torso: MLPTorso = field(default_factory=MLPTorso)
    action_head: ActionHead = field(default_factory=ActionHead)


@dataclass(frozen=True)
class MLPCriticNetworkConfig:
    pre_torso: MLPTorso = field(default_factory=MLPTorso)
    critic_head: CriticHead = field(default_factory=CriticHead)


@dataclass(frozen=True)
class MLPActorCriticConfig:
    actor_network: MLPActorNetworkConfig = field(default_factory=MLPActorNetworkConfig)
    critic_network: MLPCriticNetworkConfig = field(default_factory=MLPCriticNetworkConfig)
