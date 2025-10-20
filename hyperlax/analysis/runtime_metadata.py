"""Utilities for parsing runtime scaling experiment metadata."""

from __future__ import annotations

import json
import re
from pathlib import Path
from typing import Iterable, Optional, Tuple

import numpy as np
import pandas as pd


RUN_MODE_PATTERNS = [
    ("Sequential", re.compile(r"sequential(?:[_-]single)?", re.IGNORECASE)),
    ("Batched", re.compile(r"batched(?:[_-]?B)?(\d+)?", re.IGNORECASE)),
    (
        "Naive Vectorized",
        re.compile(r"(?:naive|ungrouped)[-_]?vectorized(?:[_-]?B)?(\d+)?", re.IGNORECASE),
    ),
    (
        "Grouped Vectorized",
        re.compile(r"grouped[_-]?vectorized(?:[_-]?B)?(\d+)?", re.IGNORECASE),
    ),
    ("Vectorized", re.compile(r"vectorized(?:[_-]?B)?(\d+)?", re.IGNORECASE)),
]

_FAMILY_TOKENS = {
    "homo_best_case": "homogeneous_best",
    "hetero_algo_hps": "hetero_algo",
    "hetero_network_hps": "hetero_network",
    "hetero_algo_and_network_hps": "hetero_algo_network",
}

__all__ = [
    "RUN_MODE_PATTERNS",
    "collect_timing_records",
    "aggregate_timing_records",
    "parse_results",
    "export_metadata_csv",
]


def _extract_mode_from_string(text: Optional[str]) -> tuple[Optional[str], Optional[int]]:
    """Find the first matching run mode within the provided text."""
    if not text:
        return None, None

    for mode_name, pattern in RUN_MODE_PATTERNS:
        match = pattern.search(text)
        if match:
            batch_size: Optional[int] = None
            if mode_name == "Sequential":
                batch_size = 1
            else:
                raw = match.group(1) if match.lastindex else None
                if raw is not None:
                    try:
                        batch_size = int(raw)
                    except ValueError:
                        batch_size = None
            return mode_name, batch_size
    return None, None


def _find_run_mode_segment(dir_parts: Iterable[str]) -> tuple[Optional[str], Optional[int], Optional[int]]:
    """Search for a dedicated path component that encodes the run mode."""
    for idx, part in enumerate(dir_parts):
        for mode_name, pattern in RUN_MODE_PATTERNS:
            if pattern.fullmatch(part):
                batch_size: Optional[int] = None
                if mode_name == "Sequential":
                    batch_size = 1
                else:
                    match = pattern.fullmatch(part)
                    if match and match.lastindex:
                        raw = match.group(1)
                        if raw is not None:
                            try:
                                batch_size = int(raw)
                            except ValueError:
                                batch_size = None
                return mode_name, batch_size, idx
    return None, None, None


def _infer_env_from_parts(parts: Iterable[str]) -> Optional[str]:
    for part in parts:
        if "gymnax_" in part or part.startswith("gymnax."):
            return part
    return None


def _config_to_exp_type_and_variant(config_id: str) -> Tuple[str, str]:
    """
    Extract a stable experiment family (exp_type) and a free-form variant suffix.

    Works regardless of extra decorations (e.g., _lower_num_envs, _no_eval, etc.).
    """
    exp_type = "unknown"
    variant = ""
    for token, etype in _FAMILY_TOKENS.items():
        idx = config_id.find(token)
        if idx != -1:
            exp_type = etype
            tail = config_id[idx + len(token) :]
            if tail.startswith("_"):
                tail = tail[1:]
            variant = tail
            break
    return exp_type, variant


_TRIMMABLE_DIR_NAMES = {
    "samples",
    "predefined_samples",
    "predefined-samples",
    "predefined_samples_eval",
    "drawn_samples",
}

_TRIMMABLE_DIR_PREFIXES = (
    "batch_",
    "optuna_exec_batch_",
    "optuna_eval_batch_",
    "optuna_suggest_batch_",
    "optuna_trial_",
)


def _is_trimmable_suffix(part: str) -> bool:
    """Return True when a trailing path segment should be ignored for config inference."""
    if part in _TRIMMABLE_DIR_NAMES:
        return True
    return any(part.startswith(prefix) for prefix in _TRIMMABLE_DIR_PREFIXES)


def _aggregate_legacy_batch_slices(df: pd.DataFrame) -> pd.DataFrame:
    """
    Backward compatibility: Manually aggregate batch slice timing files when no top-level
    timing_info.json exists (legacy batched runs before the fix).

    This function detects when we have multiple timing files from batch_XXXXX/ directories
    at the same config level without a corresponding top-level aggregated file, and creates
    a single synthetic "Batched" record that sums ALL batch slices together.
    """
    if df.empty:
        return df

    # Group by config_id to check each experiment separately
    aggregated_rows = []
    kept_rows = []

    for config_id, config_group in df.groupby("config_id", dropna=False):
        # Check if we have batch slice files (path contains batch_XXXXX pattern)
        batch_mask = config_group["relative_timing_path"].str.contains(
            r"batch_\d+/timing_info\.json$", regex=True, na=False
        )
        batch_slices = config_group[batch_mask]

        if len(batch_slices) == 0:
            # No batch slices, keep all rows as-is
            kept_rows.append(config_group)
            continue

        # Check if there's already a top-level timing_info.json (depth 0)
        top_level_mask = config_group["path_depth"] == 0
        has_top_level = top_level_mask.any()

        if has_top_level:
            # Already has top-level aggregation, keep all rows as-is
            kept_rows.append(config_group)
            continue

        # Legacy case: We have batch slices but no top-level file
        # Aggregate ALL batch slices into a SINGLE "Batched" record
        if len(batch_slices) > 1:
            timing_modes = batch_slices["timing_mode"].unique()
            print(
                f"  Aggregating {len(batch_slices)} legacy batch slices for "
                f"config_id={config_id} (modes: {', '.join(str(m) for m in timing_modes)})"
            )

            # Create single aggregated record for all batch slices
            first_row = batch_slices.iloc[0].to_dict()
            aggregated_record = {
                **first_row,
                "total_wall_time": batch_slices["total_wall_time"].sum(),
                "jit_time": batch_slices["jit_time"].sum(),
                "execution_time": batch_slices["execution_time"].sum(),
                "num_samples": batch_slices["num_samples"].sum(),
                "run_mode": "Batched",  # Unified mode
                "timing_mode": "batched",  # Unified timing mode
                "path_depth": 0,  # Synthetic top-level
                "relative_timing_path": f"{config_id}/timing_info.json (aggregated)",
                "timing_file": f"<aggregated from {len(batch_slices)} batch slices>",
                "timing_dir": first_row["timing_dir"].rsplit("/", 1)[0] if "/" in first_row["timing_dir"] else first_row["timing_dir"],
                "run_id": None,
            }

            aggregated_rows.append(aggregated_record)
        else:
            # Only one slice, keep it as-is
            kept_rows.append(batch_slices)

        # Keep non-batch rows from this config
        non_batch_rows = config_group[~batch_mask]
        if not non_batch_rows.empty:
            kept_rows.append(non_batch_rows)

    # Combine aggregated and kept rows
    result_dfs = []
    if aggregated_rows:
        result_dfs.append(pd.DataFrame(aggregated_rows))
    if kept_rows:
        result_dfs.append(pd.concat(kept_rows, ignore_index=True))

    if result_dfs:
        return pd.concat(result_dfs, ignore_index=True)
    return df


def collect_timing_records(results_root: Path) -> pd.DataFrame:
    """
    Collect timing information for every timing_info.json file under results_root.

    Returns a DataFrame with one row per timing_info.json containing both the structured
    fields extracted from the filesystem hierarchy and the timing payload itself.
    """
    rows: list[dict] = []
    timing_files = list(results_root.rglob("timing_info.json"))
    print(f"Found {len(timing_files)} timing files under {results_root}")

    for timing_file in timing_files:
        try:
            rel_parts = timing_file.relative_to(results_root).parts
        except Exception:
            continue

        dir_parts = list(rel_parts[:-1])
        full_dir_parts = dir_parts.copy()
        run_id = None
        if dir_parts and dir_parts[-1].startswith("run_"):
            run_id = dir_parts.pop()

        run_mode_segment, segment_batch_size, segment_idx = _find_run_mode_segment(dir_parts)
        if segment_idx is not None:
            # Remove the dedicated run-mode directory so config/model inference stays stable
            dir_parts.pop(segment_idx)

        # Peel off trailing container directories (e.g., samples/batch_00000) without dropping
        # the leading experiment directory that encodes the benchmark config.
        while len(dir_parts) > 1 and _is_trimmable_suffix(dir_parts[-1]):
            dir_parts.pop()

        try:
            with open(timing_file) as f:
                data = json.load(f)
        except Exception as exc:
            print(f"Warning: Failed to read {timing_file}: {exc}")
            continue

        config_id = results_root.name
        for part in dir_parts:
            if part.startswith("run_"):
                continue
            if _is_trimmable_suffix(part):
                continue
            config_id = part
            break
        model = config_id

        env = _infer_env_from_parts([*full_dir_parts, config_id, *rel_parts])
        exp_type, exp_variant = _config_to_exp_type_and_variant(config_id)

        run_mode = run_mode_segment
        batch_size: Optional[int] = segment_batch_size

        # Fallbacks if the run mode is embedded in config or payload metadata
        if run_mode is None:
            run_mode, inferred_batch = _extract_mode_from_string(config_id)
            batch_size = batch_size or inferred_batch
        if run_mode is None:
            run_mode, inferred_batch = _extract_mode_from_string(data.get("mode"))
            batch_size = batch_size or inferred_batch

        if run_mode is None:
            run_mode = "Unknown"
        if batch_size is None:
            batch_size = data.get("hparam_batch_size")
        if batch_size is None and run_mode == "Sequential":
            batch_size = 1

        rows.append(
            {
                "experiment_root": results_root.name,
                "model": model,
                "config_id": config_id,
                "exp_type": exp_type,
                "exp_variant": exp_variant,
                "run_mode": run_mode,
                "batch_size": batch_size,
                "env": env,
                "total_wall_time": data.get("total_wall_time", np.nan),
                "jit_time": data.get("jit_time", np.nan),
                "execution_time": data.get("execution_time", np.nan),
                "num_samples": data.get("num_samples", np.nan),
                "timing_mode": data.get("mode"),
                "hparam_batch_size": data.get("hparam_batch_size"),
                "run_id": run_id,
                "path_depth": max(0, len(rel_parts) - 1),
                "relative_timing_path": "/".join(rel_parts),
                "timing_file": str(timing_file),
                "timing_dir": str(timing_file.parent),
            }
        )

    if not rows:
        print("No parsable timing files found.")
        return pd.DataFrame()

    df = pd.DataFrame(rows)
    numeric_cols = ["batch_size", "total_wall_time", "jit_time", "execution_time", "num_samples", "hparam_batch_size"]
    for col in numeric_cols:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")

    # Backward compatibility: Manually aggregate batch slices when no top-level timing_info.json exists
    df = _aggregate_legacy_batch_slices(df)

    return df


def aggregate_timing_records(records: pd.DataFrame) -> pd.DataFrame:
    """
    Aggregate raw timing records into a per (model, config_id, run_mode, batch_size) summary.
    """
    if records.empty:
        return records

    if "path_depth" in records.columns:
        depth_groups = ["model", "config_id", "run_mode"]
        min_depth = records.groupby(depth_groups, dropna=False)["path_depth"].transform("min")
        records = records[records["path_depth"] == min_depth]

    group_cols = [
        "model",
        "config_id",
        "exp_type",
        "exp_variant",
        "run_mode",
        "batch_size",
        "env",
    ]
    aggregated = (
        records.groupby(group_cols, dropna=False)
        .agg(
            {
                "total_wall_time": "sum",
                "jit_time": "sum",
                "execution_time": "sum",
                "num_samples": "sum",
            }
        )
        .reset_index()
    )
    return aggregated


def parse_results(results_root: Path) -> pd.DataFrame:
    """
    Backwards-compatible helper that aggregates timing information under results_root.
    """
    records = collect_timing_records(results_root)
    if records.empty:
        return records
    return aggregate_timing_records(records)


def export_metadata_csv(records: pd.DataFrame, summary_csv_path: Path) -> Optional[Path]:
    """
    Persist experiment metadata beside the provided summary CSV.

    Returns the path to the metadata CSV if records are available, otherwise None.
    """
    if records.empty:
        return None

    metadata_path = summary_csv_path.with_name(f"{summary_csv_path.stem}_metadata.csv")
    records.to_csv(metadata_path, index=False)
    return metadata_path
