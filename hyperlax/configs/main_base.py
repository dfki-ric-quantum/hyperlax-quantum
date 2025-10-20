import dataclasses
from dataclasses import dataclass, field
from typing import Any

from hyperlax.hyperparam.tunable import Tunable


@dataclass
class BaseExperimentConfig:
    """
    The top-level configuration container for any experiment.
    It is composed of sub-configuration objects for each logical component.
    """

    algorithm: Any  # (e.g., PPOConfig, DQNConfig).
    env: Any  # BaseEnvironmentConfig
    training: Any  # BaseTrainingConfig
    logger: Any  # LoggerConfig
    config_name: str
    git_tag: str = ""
    experiment_tags: str = ""
    experiment_mode: str = "batched"
    _base_module_path_: Any = field(default=None, repr=False)


def get_sampling_distributions_from_config(config_obj: Any) -> dict[str, Any]:
    """
    Recursively traverses a config object to find all tunable fields and extract their distributions.
    """
    distributions = {}

    def _recursive_find(obj: Any, path_prefix: str) -> None:
        if not dataclasses.is_dataclass(obj):
            return
        for f in dataclasses.fields(obj):
            # Avoid recursing into private-like fields
            if f.name.startswith("_"):
                continue

            field_value = getattr(obj, f.name)
            new_path = f"{path_prefix}.{f.name}" if path_prefix else f.name

            if isinstance(field_value, Tunable):
                if not field_value.is_fixed and field_value.distribution:
                    distributions[new_path] = field_value.distribution
            elif dataclasses.is_dataclass(field_value):
                _recursive_find(field_value, new_path)

    # Start the search from the 'algorithm' component, as this is where all tunable params reside
    if hasattr(config_obj, "algorithm"):
        _recursive_find(config_obj.algorithm, "algorithm")
    return distributions
