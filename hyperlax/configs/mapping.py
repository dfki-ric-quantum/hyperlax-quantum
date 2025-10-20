import logging
from dataclasses import fields, is_dataclass, replace
from typing import Any, TypeVar

from hyperlax.hyperparam.tunable import Tunable
from hyperlax.utils.type_cast import cast_value_to_expected_type

logger = logging.getLogger(__name__)

T = TypeVar("T")


def _apply_nested_updates(config_obj: T, updates: dict[str, Any]) -> T:
    """
    Recursively applies updates from a nested dictionary to a nested dataclass.
    This is an immutable operation; it returns a new object.
    It now handles updating the .value attribute of Tunable objects.
    """
    if not is_dataclass(config_obj) or not updates:
        return config_obj

    replacements: dict[str, Any] = {}

    for key, value in updates.items():
        if not hasattr(config_obj, key):
            continue

        field_value = getattr(config_obj, key)

        if isinstance(value, dict) and is_dataclass(field_value):
            # Recurse for nested updates
            updated_nested_obj = _apply_nested_updates(field_value, value)
            if updated_nested_obj is not field_value:
                replacements[key] = updated_nested_obj
        elif isinstance(field_value, Tunable):
            # Update the value inside the Tunable object
            # Use the existing type for casting
            casted_value = cast_value_to_expected_type(value, field_value.expected_type)
            replacements[key] = replace(field_value, value=casted_value)
        else:
            # Direct update for non-dataclass, non-tunable fields
            replacements[key] = value

    if not replacements:
        return config_obj

    return replace(config_obj, **replacements)


def update_config_from_flat_dict(
    config_obj: T, flat_params: dict[str, Any], root_path_prefix: str = ""
) -> T:
    """
    Updates a dataclass instance with values from a flat dictionary.
    This is an immutable operation; it returns a new, updated object.
    """
    nested_updates: dict[str, Any] = {}
    for flat_key, value in flat_params.items():
        if not flat_key.startswith(root_path_prefix):
            continue

        relative_path = flat_key[len(root_path_prefix) :]
        path_parts = relative_path.split(".")

        # Convert the flat key path into a nested dictionary structure
        current_level = nested_updates
        for part in path_parts[:-1]:
            current_level = current_level.setdefault(part, {})
        current_level[path_parts[-1]] = value

    if not nested_updates:
        return config_obj

    return _apply_nested_updates(config_obj, nested_updates)


def _extract_tunable_values_to_flat_dict_recursive(
    obj: Any,
    current_relative_path: str,
    root_path_prefix: str,
    result_dict: dict[str, Any],
) -> None:
    """
    Recursively extracts values of fields marked with metadata={'tunable': True}
    from a dataclass into a flat dictionary.
    """
    if not is_dataclass(obj):
        return

    for field_obj in fields(obj):
        field_name = field_obj.name
        if field_name.startswith("_"):
            continue

        relative_config_path = (
            f"{current_relative_path}.{field_name}" if current_relative_path else field_name
        )
        full_flat_key = f"{root_path_prefix}{relative_config_path}"

        field_value = getattr(obj, field_name)

        if field_obj.metadata.get("tunable", False):
            result_dict[full_flat_key] = field_value

        if is_dataclass(field_value):
            _extract_tunable_values_to_flat_dict_recursive(
                field_value, relative_config_path, root_path_prefix, result_dict
            )


def get_tunable_values_as_flat_dict(config_obj: Any, root_path_prefix: str = "") -> dict[str, Any]:
    """
    Extracts all tunable parameter values from a dataclass instance into a flat dictionary.

    Args:
        config_obj: The dataclass instance to extract values from (e.g., exp_config.network).
        root_path_prefix: A prefix to prepend to the flat keys (e.g., "network." or "hyperparam.").

    Returns:
        A flat dictionary where keys are full flat paths (e.g., "network.actor_network.pre_torso.layer_sizes")
        and values are the corresponding parameter values from the config_obj.
    """
    result = {}
    _extract_tunable_values_to_flat_dict_recursive(config_obj, "", root_path_prefix, result)
    return result


def _apply_distributions_recursive(obj: Any, path_prefix: str, dist_dict: dict[str, Any]) -> Any:
    if not is_dataclass(obj):
        return obj

    replacements = {}
    for f in fields(obj):
        field_name = f.name
        new_path = f"{path_prefix}.{f.name}" if path_prefix else field_name
        field_value = getattr(obj, f.name)

        if isinstance(field_value, Tunable) and new_path in dist_dict:
            # Found a match! Create a new Tunable with the distribution.
            new_tunable = replace(field_value, distribution=dist_dict[new_path], is_fixed=False)
            replacements[field_name] = new_tunable
        elif is_dataclass(field_value):
            # Recurse
            updated_sub_obj = _apply_distributions_recursive(field_value, new_path, dist_dict)
            if updated_sub_obj is not field_value:
                replacements[field_name] = updated_sub_obj

    if not replacements:
        return obj
    return replace(obj, **replacements)


def apply_distributions_to_config(config: T, dist_dict: dict[str, Any]) -> T:
    """
    Applies distributions from a flat dictionary to the `Tunable` fields
    in a BaseExperimentConfig object. Starts the process at the `algorithm` level.
    """
    if not hasattr(config, "algorithm"):
        return config

    updated_algorithm_config = _apply_distributions_recursive(
        config.algorithm, "algorithm", dist_dict
    )
    # logger.debug(f"After dist applied to config {updated_algorithm_config}")
    return replace(config, algorithm=updated_algorithm_config)
