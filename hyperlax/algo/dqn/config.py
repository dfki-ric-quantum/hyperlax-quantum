from dataclasses import dataclass, field

from hyperlax.algo.dqn.hyperparam import DQNHyperparams
from hyperlax.configs.network.critic_drpqc import DRPQCQNetworkConfig
from hyperlax.configs.network.critic_mlp import MLPQNetworkConfig
from hyperlax.configs.network.critic_tmlp import TMLPQNetworkConfig


@dataclass
class DQNConfig:
    """Component config for the DQN algorithm."""

    _target_: str = "hyperlax.algo.dqn.main_dqn.main"
    network: MLPQNetworkConfig | DRPQCQNetworkConfig | TMLPQNetworkConfig = field(
        default_factory=MLPQNetworkConfig
    )
    hyperparam: DQNHyperparams = field(default_factory=DQNHyperparams)
