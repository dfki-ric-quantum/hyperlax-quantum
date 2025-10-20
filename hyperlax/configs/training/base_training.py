from dataclasses import dataclass, field
from typing import Any


@dataclass(frozen=True)
class BaseTrainingConfig:
    seed: int = 42
    device: str = "cuda"
    num_devices: int = -1
    devices: list[str] | None = None
    total_timesteps: int = int(1e5)
    update_batch_size: int = 1
    num_envs: int = -1
    num_updates: int = -1
    num_updates_per_eval: int = -1
    evaluation_greedy: bool = False
    num_eval_episodes: int = 16
    num_evaluation: int = 5
    absolute_metric: bool = False
    num_agents_slash_seeds: int = 8
    max_batchable_chunk_size: int = 1
    jit_enabled: bool = True
    test_mode: bool = False
    # normalize_observations: bool = True
    normalize_method: str = "running_meanstd"
    hyperparam_batch_enabled: bool = False
    hyperparam_batch_samples: dict[str, list[list[Any]]] = field(default_factory=dict)
    hyperparam_batch_size: int = -1
    hyperparam_batch_sample_ids: list[int] = field(default_factory=lambda: [])
    launch_context: str = "imported"  # "main" or "imported"
    trainer_style: str = "phased"  # "phased"/"phased_single_step" -> faster!
