"""Utility functions for hyperparameter experiments."""

import typing
from dataclasses import asdict, is_dataclass
from enum import Enum
from typing import Any

import numpy as np
import yaml


# Helper to stringify generic types like List[int]
def stringify_generic_type(data: Any) -> str:
    origin = typing.get_origin(data)
    args = typing.get_args(data)
    if origin and args:
        args_str = ", ".join(
            (
                stringify_generic_type(arg)
                if hasattr(arg, "__origin__") or isinstance(arg, type)
                else repr(arg)
            )
            for arg in args
        )
        return f"{origin.__name__}[{args_str}]"
    elif isinstance(data, type):
        return data.__name__
    return repr(data)  # Fallback for other unhandled types


class ExperimentEncoder(yaml.SafeDumper):
    """Custom YAML encoder with general handling of custom types."""

    def represent_data(self, data):
        """General representation handling for custom types."""
        if isinstance(data, Enum):
            # Handle any Enum type by using its value
            return self.represent_scalar("tag:yaml.org,2002:str", data.value)
        # #elif isinstance(data, Type) and isinstance(data, list[int]):
        # elif isinstance(data, Type) and isinstance(data, list[int]):
        #     return self.represent_scalar('tag:yaml.org,2002:str', "list[int]")
        # elif isinstance(data, Type): # Check if data is a type object itself
        #     return self.represent_scalar('tag:yaml.org,2002:str', data.__name__)
        elif typing.get_origin(data) is not None:
            # It's a generic type like List[int], Dict[str, float], etc.
            # We want to represent it as a string like "List[int]"
            return self.represent_scalar("tag:yaml.org,2002:str", stringify_generic_type(data))
        elif isinstance(data, type):  # Handles simple types like <class 'float'>, <class 'list'>
            return self.represent_scalar("tag:yaml.org,2002:str", data.__name__)
        elif is_dataclass(data):
            # Handle any dataclass by converting to dict
            return self.represent_dict(asdict(data))
        elif isinstance(data, np.floating):
            # Handle numpy float types
            return self.represent_float(float(data))
        elif isinstance(data, np.integer):
            # Handle numpy int types
            return self.represent_scalar("tag:yaml.org,2002:int", str(int(data)))

        # Fall back to default representation
        return super().represent_data(data)


def save_experiment_config(config: Any, filepath: str):
    """Save experiment config with automatic type handling."""
    # Convert config to dict if it's a dataclass
    if is_dataclass(config):
        config_dict = asdict(config)
    else:
        config_dict = config

    with open(filepath, "w") as f:
        yaml.dump(config_dict, f, Dumper=ExperimentEncoder, default_flow_style=False)


def load_experiment_config(filepath: str) -> dict:
    """Load experiment config from YAML file."""
    with open(filepath) as f:
        return yaml.safe_load(f)
