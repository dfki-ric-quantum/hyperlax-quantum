from dataclasses import dataclass, field

from hyperlax.configs.env.base_env import BaseEnvironmentConfig


@dataclass(frozen=True)
class ScenarioConfig:
    name: str = "swimmer"
    task_name: str = "swimmer"


@dataclass(frozen=True)
class BraxSwimmerConfig(BaseEnvironmentConfig):
    env_name: str = "brax"
    scenario: ScenarioConfig = field(default_factory=ScenarioConfig)
    kwargs: dict[str, str] = field(default_factory=lambda: {"backend": "generalized"})
    eval_metric: str = "episode_return"
