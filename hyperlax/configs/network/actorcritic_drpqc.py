from dataclasses import dataclass, field

from hyperlax.hyperparam.distributions import Categorical
from hyperlax.hyperparam.tunable import Tunable


@dataclass(frozen=True)
class DataReuploadingParametrizedQuantumCircuit:
    _target_: str = "hyperlax.network.data_reuploading.DataReuploadingTorso"
    n_features: int = -1
    n_layers: Tunable = field(
        default_factory=lambda: Tunable(
            value=5,
            is_vectorized=False,
            is_fixed=False,
            distribution=Categorical(values=[1, 2, 5, 10]),
            expected_type=int,
        )
    )
    n_vstack: Tunable = field(
        default_factory=lambda: Tunable(
            value=1,
            is_vectorized=False,
            is_fixed=False,
            distribution=Categorical(values=[1, 2, 3, 4, 5]),
            expected_type=int,
        )
    )
    scaling: float = 1.0
    # scaling: Tunable = field(default_factory=lambda: Tunable(
    #     value=1.0, is_vectorized=False, is_fixed=False,
    #     distribution=UniformContinuous(domain=(0.1, 2.0)), expected_type=float
    # ))
    observable_type: str = "full"
    max_vmap: int | None = None
    jit: bool = False
    dev_type: str = "default.qubit.jax"
    draw_circuit: bool = False


@dataclass(frozen=True)
class ActionHead:
    _target_: str = "hyperlax.network.heads.NormalAffineTanhDistributionHead"


@dataclass(frozen=True)
class CriticHead:
    _target_: str = "hyperlax.network.heads.ScalarCriticHead"


@dataclass(frozen=True)
class DRPQCActorNetworkConfig:
    pre_torso: DataReuploadingParametrizedQuantumCircuit = field(
        default_factory=DataReuploadingParametrizedQuantumCircuit
    )
    action_head: ActionHead = field(default_factory=ActionHead)


@dataclass(frozen=True)
class DRPQCCriticNetworkConfig:
    pre_torso: DataReuploadingParametrizedQuantumCircuit = field(
        default_factory=DataReuploadingParametrizedQuantumCircuit
    )
    critic_head: CriticHead = field(default_factory=CriticHead)


@dataclass(frozen=True)
class DRPQCActorCriticConfig:
    actor_network: DRPQCActorNetworkConfig = field(default_factory=DRPQCActorNetworkConfig)
    critic_network: DRPQCCriticNetworkConfig = field(default_factory=DRPQCCriticNetworkConfig)
