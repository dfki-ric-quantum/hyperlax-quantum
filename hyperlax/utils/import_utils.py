import importlib.util
import sys
from pathlib import Path
from types import ModuleType


def import_config_from_path(config_path: str) -> ModuleType:
    """
    Imports a Python module from a given file path.

    This is useful for dynamically loading configuration files
    that are not part of the standard Python package structure.

    Args:
        config_path: The file path to the Python module (e.g., 'configs/my_experiment.py').

    Returns:
        A module object representing the imported file.

    Raises:
        FileNotFoundError: If the config_path does not exist.
        ImportError: If there's an issue importing the module.
    """
    config_file_path = Path(config_path).resolve()

    if not config_file_path.exists():
        raise FileNotFoundError(f"Configuration file not found: {config_file_path}")

    # Create a unique module name from the file path
    module_name = f"dynamic_config_{config_file_path.stem}_{config_file_path.parent.name}"

    spec = importlib.util.spec_from_file_location(module_name, config_file_path)
    if spec is None:
        raise ImportError(f"Could not create module spec for {config_file_path}")

    module = importlib.util.module_from_spec(spec)
    sys.modules[module_name] = module

    try:
        spec.loader.exec_module(module)
    except Exception as e:
        raise ImportError(f"Error executing module {config_file_path}: {e}") from e

    return module
