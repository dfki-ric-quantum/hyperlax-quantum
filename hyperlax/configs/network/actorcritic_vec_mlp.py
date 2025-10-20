from dataclasses import dataclass, field

from hyperlax.hyperparam.distributions import Categorical, UniformDiscrete
from hyperlax.hyperparam.tunable import Tunable


@dataclass(frozen=True)
class MLPTorso:
    """Contains only the constructor arguments for the ParametricMLPTorso."""

    _target_: str = "hyperlax.network.parametric_torso.ParametricMLPTorso"
    max_depth: int = 4
    max_width: int = 256
    input_dim: int = -1  # Patched by runner


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
    """Config for the actor network, containing both the torso's constructor args
    and the tunable architectural parameters."""

    pre_torso: MLPTorso = field(default_factory=MLPTorso)
    action_head: ActionHead = field(default_factory=ActionHead)  # Assumes ActionHead is defined

    # Vectorized Architectural Hyperparameters live here
    num_layers: Tunable = field(
        default_factory=lambda: Tunable(
            value=2,
            is_vectorized=True,
            is_fixed=False,
            distribution=UniformDiscrete(domain=(1, 4)),
            expected_type=int,
        )
    )
    width: Tunable = field(
        default_factory=lambda: Tunable(
            value=128,
            is_vectorized=True,
            is_fixed=False,
            distribution=Categorical(values=[64, 128, 256]),
            expected_type=int,
        )
    )
    use_layer_norm: Tunable = field(
        default_factory=lambda: Tunable(
            value=False,
            is_vectorized=True,
            is_fixed=False,
            distribution=Categorical(values=[True, False]),
            expected_type=bool,
        )
    )
    activation: Tunable = field(
        default_factory=lambda: Tunable(
            value="silu",
            is_vectorized=True,
            is_fixed=False,
            distribution=Categorical(values=["relu", "tanh", "silu"]),
            expected_type=str,
        )
    )


@dataclass(frozen=True)
class MLPCriticNetworkConfig:
    """Config for the critic network."""

    pre_torso: MLPTorso = field(default_factory=MLPTorso)
    critic_head: CriticHead = field(default_factory=CriticHead)  # Assumes CriticHead is defined

    # Add its own tunables here if you want them to be different from the actor
    num_layers: Tunable = field(
        default_factory=lambda: Tunable(
            value=2,
            is_vectorized=True,
            is_fixed=False,
            distribution=UniformDiscrete(domain=(1, 4)),
            expected_type=int,
        )
    )
    width: Tunable = field(
        default_factory=lambda: Tunable(
            value=128,
            is_vectorized=True,
            is_fixed=False,
            distribution=Categorical(values=[64, 128, 256]),
            expected_type=int,
        )
    )
    use_layer_norm: Tunable = field(
        default_factory=lambda: Tunable(
            value=False,
            is_vectorized=True,
            is_fixed=False,
            distribution=Categorical(values=[True, False]),
            expected_type=bool,
        )
    )
    activation: Tunable = field(
        default_factory=lambda: Tunable(
            value="silu",
            is_vectorized=True,
            is_fixed=False,
            distribution=Categorical(values=["relu", "tanh", "silu"]),
            expected_type=str,
        )
    )


@dataclass(frozen=True)
class MLPActorCriticConfig:
    # Both networks will now point to the same MLPTorso config object
    # by default, effectively sharing the architectural HPs.
    actor_network: MLPActorNetworkConfig = field(default_factory=MLPActorNetworkConfig)
    critic_network: MLPCriticNetworkConfig = field(default_factory=MLPCriticNetworkConfig)
