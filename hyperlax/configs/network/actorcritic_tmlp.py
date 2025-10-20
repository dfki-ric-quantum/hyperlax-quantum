from dataclasses import dataclass, field

from hyperlax.hyperparam.distributions import Categorical
from hyperlax.hyperparam.tunable import Tunable

# hidden_dim and num_nodes_to_decompose are not independent
# network's hidden_dim is constrained by its num_nodes_to_decompose.


@dataclass(frozen=True)
class MPODecomposedMLPTorso:
    _target_: str = "hyperlax.network.tensorized_mlp.TMLPwithMPO"
    in_dim: int = 0
    hidden_dim: Tunable = field(
        default_factory=lambda: Tunable(
            value=256,
            is_vectorized=False,
            is_fixed=True,  # This parameter is now controlled by a joint sampler
            distribution=None,
            expected_type=int,
        )
    )
    num_nodes_to_decompose: Tunable = field(
        default_factory=lambda: Tunable(
            value=4,
            is_vectorized=False,
            is_fixed=True,  # This parameter is now controlled by a joint sampler
            distribution=None,
            expected_type=int,
        )
    )
    bond_dim: Tunable = field(
        default_factory=lambda: Tunable(
            value=8,
            is_vectorized=False,
            is_fixed=False,
            distribution=Categorical(values=[4, 8, 16]),
            expected_type=int,
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
    num_mpo_layers: Tunable = field(
        default_factory=lambda: Tunable(
            value=1,
            is_vectorized=False,
            is_fixed=False,
            distribution=Categorical(values=[1, 2]),
            expected_type=int,
        )
    )
    out_dim: int = 256
    arch_choice: Tunable = field(  # This is a proxy for joint sampling
        default_factory=lambda: Tunable(
            value=0, is_vectorized=False, is_fixed=True, expected_type=int
        )
    )


@dataclass(frozen=True)
class ActionHead:
    _target_: str = "hyperlax.network.heads.NormalAffineTanhDistributionHead"


@dataclass(frozen=True)
class CriticHead:
    _target_: str = "hyperlax.network.heads.ScalarCriticHead"


@dataclass(frozen=True)
class TMLPMPOActorNetworkConfig:
    pre_torso: MPODecomposedMLPTorso = field(default_factory=MPODecomposedMLPTorso)
    action_head: ActionHead = field(default_factory=ActionHead)


@dataclass(frozen=True)
class TMLPMPOCriticNetworkConfig:
    pre_torso: MPODecomposedMLPTorso = field(default_factory=MPODecomposedMLPTorso)
    critic_head: CriticHead = field(default_factory=CriticHead)


@dataclass(frozen=True)
class TMLPMPODActorCriticConfig:
    actor_network: TMLPMPOActorNetworkConfig = field(default_factory=TMLPMPOActorNetworkConfig)
    critic_network: TMLPMPOCriticNetworkConfig = field(default_factory=TMLPMPOCriticNetworkConfig)
