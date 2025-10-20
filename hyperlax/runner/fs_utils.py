import datetime
import gc
import json
import logging
import subprocess
from dataclasses import asdict
from pathlib import Path
from typing import Any

import jax
from jax.lib import xla_bridge

logger = logging.getLogger(__name__)


def save_pip_packages(
    output_file: str, freeze: bool = True, print_to_console: bool = True
) -> None:
    """Save pip package list to file."""
    try:
        command = "pip freeze" if freeze else "pip list"
        result = subprocess.run(command.split(), capture_output=True, text=True, check=True)
        if print_to_console:
            logger.debug(result.stdout)
        with open(output_file, "w") as f:
            f.write(result.stdout)
        logger.info(f"Packages saved to {output_file}")
    except (subprocess.CalledProcessError, FileNotFoundError) as e:
        logger.error(f"Failed to save pip packages: {e}")


def get_completed_runs(output_dir: str | Path) -> set[int]:
    """Get set of completed run IDs from an output directory."""
    sample_dir = Path(output_dir)
    if not sample_dir.exists():
        return set()

    completed_runs = set()
    for run_dir in sample_dir.glob("run_*"):
        try:
            if (run_dir / "success.txt").exists():
                run_id = int(run_dir.name.split("_")[1])
                completed_runs.add(run_id)
        except (ValueError, IndexError):
            continue
    return completed_runs


def clear_memory() -> None:
    """Clear Python garbage collector and JAX memory."""
    try:
        gc.collect()
        jax.clear_caches()
        backend = xla_bridge.get_backend()
        for device in backend.devices():
            if device.platform in ("cuda", "gpu"):
                jax.device_put(0, device)  # Synchronize
    except Exception as e:
        logger.error(f"Error during memory clearing: {e}")


def clear_memory_between_runs(func):
    """Decorator to clear memory before and after each experiment run."""

    def wrapper(*args, **kwargs):
        logger.debug("Clearing memory before run...")
        clear_memory()
        try:
            return func(*args, **kwargs)
        finally:
            logger.debug("Clearing memory after run...")
            clear_memory()

    return wrapper


def save_args_config_and_metadata(
    args_config: Any, output_dir: Path, save_requirement_txt: bool = False
) -> None:
    """Saves CLI arguments and package versions."""
    # output_dir = Path(args_config.output_dir)

    # Save arguments
    args_path = output_dir / "args.json"
    args_dict_to_save = asdict(args_config)
    try:
        with open(args_path, "w") as f:
            json.dump(args_dict_to_save, f, indent=2, default=str)
        logger.info(f"Saved cli arguments to: {args_path}")
    except Exception as e:
        logger.error(f"Error saving arguments to {args_path}: {e}")

    if save_requirement_txt:
        # Save package versions
        req_path = output_dir / "requirements.txt"
        save_pip_packages(str(req_path), print_to_console=False)


def mark_experiment_complete(output_path: str | Path) -> None:
    """Marks an experiment as complete by creating a success.txt file."""
    success_file = Path(output_path) / "success.txt"
    with open(success_file, "w") as f:
        f.write(f"Success\nCompleted at: {datetime.datetime.now().isoformat()}\n")
    logger.info(f"Marked {output_path} as complete.")


def is_experiment_complete(output_path: str | Path) -> bool:
    """Checks if an experiment is complete."""
    return (Path(output_path) / "success.txt").exists()
