import csv
import json
import logging
from collections import defaultdict
from pathlib import Path
from typing import Any

import pandas as pd

from hyperlax.configs.main_base import BaseExperimentConfig
from hyperlax.hyperparam.base_types import flatten_tunables
from hyperlax.runner.batch_utils import find_sample_id_key
from hyperlax.utils.type_cast import cast_value_to_expected_type

logger = logging.getLogger(__name__)


def save_samples(
    output_dir: str | Path,
    samples_unnormalized: dict | list | tuple,
    samples_normalized: dict | list | tuple | None = None,
    experiment_type: str = "independent_samples",
):
    """
    Save samples to files using pandas DataFrame with 2D array structure.
    """
    SUPPORTED_TYPES = {"sobol_matrices_A_B_AB", "independent_samples", "loaded_file"}
    if experiment_type not in SUPPORTED_TYPES:
        raise ValueError(
            f"Unsupported experiment type: {experiment_type}. Must be one of: {SUPPORTED_TYPES}"
        )

    if not samples_unnormalized:
        raise ValueError("samples_unnormalized cannot be None or empty")

    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    def dict_to_df(d):
        if not isinstance(d, dict):
            raise ValueError(f"Expected dictionary, got {type(d)}")
        try:
            return pd.DataFrame.from_dict(d)
        except Exception as e:
            logger.warning(f"Warning: Failed to convert dictionary to DataFrame: {e}")
            return pd.DataFrame()

    def reorder_df_columns(df: pd.DataFrame, sample_id_key: str) -> pd.DataFrame:
        if sample_id_key in df.columns:
            cols = list(df.columns)
            cols.remove(sample_id_key)
            return df[[sample_id_key] + cols]
        return df

    # Find the dynamic sample_id key to use for column reordering
    sample_id_key_in_data = "sample_id"  # Default for user files
    if isinstance(samples_unnormalized, dict) and any(
        k.endswith(".sample_id") for k in samples_unnormalized.keys()
    ):
        sample_id_key_in_data = next(
            k for k in samples_unnormalized.keys() if k.endswith(".sample_id")
        )
    elif (
        isinstance(samples_unnormalized, (list, tuple))
        and isinstance(samples_unnormalized[0], dict)
        and any(k.endswith(".sample_id") for k in samples_unnormalized[0].keys())
    ):
        sample_id_key_in_data = next(
            k for k in samples_unnormalized[0].keys() if k.endswith(".sample_id")
        )

    if experiment_type == "sobol_matrices_A_B_AB":
        (output_dir / "sample_A").mkdir(exist_ok=True)
        (output_dir / "sample_B").mkdir(exist_ok=True)

        if not isinstance(samples_unnormalized, (list, tuple)) or len(samples_unnormalized) < 2:
            raise ValueError("Sobol matrices require at least A and B samples as a list/tuple")

        df_A_unn = reorder_df_columns(dict_to_df(samples_unnormalized[0]), sample_id_key_in_data)
        df_B_unn = reorder_df_columns(dict_to_df(samples_unnormalized[1]), sample_id_key_in_data)

        df_A_unn.to_csv(output_dir / "sample_A" / "unnormalized.csv", index=False)
        df_B_unn.to_csv(output_dir / "sample_B" / "unnormalized.csv", index=False)

        if samples_normalized and len(samples_normalized) >= 2:
            df_A_norm = reorder_df_columns(
                dict_to_df(samples_normalized[0]), sample_id_key_in_data
            )
            df_B_norm = reorder_df_columns(
                dict_to_df(samples_normalized[1]), sample_id_key_in_data
            )
            df_A_norm.to_csv(output_dir / "sample_A" / "normalized.csv", index=False)
            df_B_norm.to_csv(output_dir / "sample_B" / "normalized.csv", index=False)

        if len(samples_unnormalized) == 3 and samples_unnormalized[2]:
            ab_dir = output_dir / "sample_AB"
            ab_dir.mkdir(exist_ok=True)
            for param_name, ab_dict in samples_unnormalized[2]:
                df = reorder_df_columns(dict_to_df(ab_dict), sample_id_key_in_data)
                df.to_csv(ab_dir / f"unnormalized_{param_name}.csv", index=False)

            if samples_normalized and len(samples_normalized) == 3:
                for param_name, ab_dict in samples_normalized[2]:
                    df = reorder_df_columns(dict_to_df(ab_dict), sample_id_key_in_data)
                    df.to_csv(ab_dir / f"normalized_{param_name}.csv", index=False)
    else:
        sample_dir = output_dir / "samples"
        sample_dir.mkdir(exist_ok=True)
        df_unn = reorder_df_columns(dict_to_df(samples_unnormalized), sample_id_key_in_data)
        df_unn.to_csv(sample_dir / "unnormalized.csv", index=False)

        if samples_normalized:
            df_norm = reorder_df_columns(dict_to_df(samples_normalized), sample_id_key_in_data)
            df_norm.to_csv(sample_dir / "normalized.csv", index=False)

    logger.info(f"Samples saved successfully to {output_dir}")


def load_csv(file_path: str) -> dict[str, list[Any]]:
    """Loads hyperparameter combinations from a CSV file."""
    data = defaultdict(list)
    try:
        with open(file_path, newline="") as csvfile:
            reader = csv.DictReader(csvfile)
            headers = reader.fieldnames
            if not headers:
                raise ValueError(f"CSV file '{file_path}' is empty or missing headers.")
            for row in reader:
                for header in headers:
                    data[header].append(row[header])
    except FileNotFoundError:
        raise FileNotFoundError(f"Hyperparameter file not found: {file_path}")
    return dict(data)


def load_json(file_path: str) -> dict[str, list[Any]]:
    """Loads hyperparameter combinations from a JSON file."""
    data = defaultdict(list)
    try:
        with open(file_path) as f:
            loaded_data = json.load(f)

        if not isinstance(loaded_data, list):
            raise ValueError("JSON file must contain a list of hyperparameter dictionaries.")
        if not loaded_data:
            raise ValueError("JSON file is empty.")

        headers = list(loaded_data[0].keys())
        for item in loaded_data:
            for header in headers:
                data[header].append(item.get(header))
    except FileNotFoundError:
        raise FileNotFoundError(f"Hyperparameter file not found: {file_path}")
    return dict(data)


def _cast_hyperparam_dict_to_expected_types(
    hyperparam_sample: dict[str, list[Any]],
    base_config: BaseExperimentConfig,
) -> dict[str, list[Any]]:
    """Cast hyperparameters to their expected types according to spec."""
    casted_samples = {}
    flat_tunables = flatten_tunables(base_config)

    for param_name, values in hyperparam_sample.items():
        # sample_id is now handled by the dynamic key, so we cast it like any other param
        tunable_spec = flat_tunables.get(param_name)
        if tunable_spec is None:
            # This handles cases where a file might have extra columns not in the config.
            logger.warning(
                f"No Tunable spec found for parameter '{param_name}' from file. Values will be kept as string."
            )
            casted_samples[param_name] = values
            continue

        try:
            casted_values = [
                cast_value_to_expected_type(val, tunable_spec.expected_type) for val in values
            ]
            casted_samples[param_name] = casted_values
        except TypeError as e:
            logger.warning(
                f"TypeError during casting for parameter '{param_name}'. Keeping original values. Error: {e}"
            )
            casted_samples[param_name] = values

    return casted_samples


def load_hyperparams_from_file(
    file_path: str,
    expected_params: list[str],
    base_config: BaseExperimentConfig,
    with_sample_id: bool = True,
) -> dict[str, list[Any]]:
    """Load hyperparameters from file and validate against expected params."""
    file_ext = Path(file_path).suffix.lower()

    if file_ext == ".csv":
        loaded_data = load_csv(file_path)
    elif file_ext == ".json":
        loaded_data = load_json(file_path)
    else:
        raise ValueError(f"Unsupported file format: {file_ext}. Use .csv or .json.")

    # Find the dynamic sample_id key from the config
    sample_id_key_full_path = find_sample_id_key(base_config)

    num_loaded = len(next(iter(loaded_data.values()), []))

    # Check if a simple 'sample_id' column exists from the file
    if "sample_id" in loaded_data:
        # If it does, move its data to the full path key and remove the simple key
        if sample_id_key_full_path not in loaded_data:
            loaded_data[sample_id_key_full_path] = loaded_data[
                "sample_id"
            ]  # NOTE this was the first version might want to keep this for a some time?
        del loaded_data["sample_id"]
    elif with_sample_id and num_loaded > 0:
        # If no sample_id column exists at all, generate it using the full path key
        logger.info(f"Generating sequential '{sample_id_key_full_path}' for loaded file.")
        loaded_data[sample_id_key_full_path] = list(range(num_loaded))

    # Validate and filter keys
    loaded_keys = set(loaded_data.keys())
    # The expected keys from flatten_tunables are already full paths.
    expected_keys_set = set(expected_params)
    if with_sample_id:
        expected_keys_set.add(sample_id_key_full_path)

    missing_keys = expected_keys_set - loaded_keys
    if missing_keys:
        logger.warning(
            f"Expected parameters not found in file (will use defaults): {missing_keys}"
        )

    unexpected_keys = loaded_keys - expected_keys_set
    if unexpected_keys:
        logger.warning(f"Unexpected parameters found in file (will be ignored): {unexpected_keys}")
        loaded_data = {k: v for k, v in loaded_data.items() if k in expected_keys_set}

    if not loaded_data:
        raise ValueError(f"No valid parameters loaded from '{file_path}'")

    return _cast_hyperparam_dict_to_expected_types(loaded_data, base_config)
