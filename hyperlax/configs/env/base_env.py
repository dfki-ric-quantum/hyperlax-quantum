from dataclasses import dataclass, field
from typing import Any


@dataclass(frozen=True)
class BaseEnvironmentConfig:
    env_name: str = ""
    scenario: Any = None
    kwargs: dict[str, Any] = field(default_factory=dict)
    eval_metric: str = "episode_return"
    obs_dim: float = -1
    act_dim: float = -1
    act_minimum: float = -1
    act_maximum: float = 1
