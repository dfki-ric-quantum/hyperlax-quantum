from dataclasses import dataclass, field

from hyperlax.configs.env.base_env import BaseEnvironmentConfig


@dataclass(frozen=True)
class ScenarioConfig:
    name: str = "MountainCarContinuous-v0"
    task_name: str = "mountain_car_continuous"


@dataclass(frozen=True)
class GymnaxMountainCarContinuousConfig(BaseEnvironmentConfig):
    env_name: str = "gymnax"
    scenario: ScenarioConfig = field(default_factory=ScenarioConfig)
    kwargs: dict = field(default_factory=dict)
    eval_metric: str = "episode_return"
