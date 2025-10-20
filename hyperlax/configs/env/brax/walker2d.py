from dataclasses import dataclass, field

from hyperlax.configs.env.base_env import BaseEnvironmentConfig


@dataclass(frozen=True)
class ScenarioConfig:
    name: str = "walker2d"
    task_name: str = "walker2d"


@dataclass(frozen=True)
class BraxWalker2dConfig(BaseEnvironmentConfig):
    env_name: str = "brax"
    scenario: ScenarioConfig = field(default_factory=ScenarioConfig)
    kwargs: dict[str, str] = field(default_factory=lambda: {"backend": "positional"})
    eval_metric: str = "episode_return"
