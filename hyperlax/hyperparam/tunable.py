import dataclasses
from dataclasses import dataclass, fields, replace
from typing import Any

from hyperlax.hyperparam.distributions import BaseDistribution


@dataclass(frozen=True)
class Tunable:
    """A container for a hyperparameter that holds its value and specification."""

    value: Any
    is_vectorized: bool
    is_fixed: bool = True
    expected_type: type = Any
    distribution: BaseDistribution | None = None
    help: str = ""

    def __post_init__(self):
        # Validation logic can go here if needed
        if not self.is_fixed and self.distribution is None:
            # We allow this for cases where values are loaded from a file
            pass
        if self.is_fixed and self.distribution is not None:
            raise ValueError("Fixed parameter cannot have a distribution.")


def get_tunable_values(config_obj: Any) -> Any:
    """
    Recursively replaces Tunable fields in a dataclass with their .value attribute.
    Returns a new dataclass instance with primitive types.
    """
    if not dataclasses.is_dataclass(config_obj):
        return config_obj

    replacements = {}
    for f in fields(config_obj):
        field_value = getattr(config_obj, f.name)
        if isinstance(field_value, Tunable):
            replacements[f.name] = field_value.value
        elif dataclasses.is_dataclass(field_value):
            replacements[f.name] = get_tunable_values(field_value)

    return replace(config_obj, **replacements)
