"""Integration helpers for Optuna studies -> custom plots."""

from __future__ import annotations

import logging
import math
import shutil
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, List

import numpy as np
import optuna
import pandas as pd
import plotly.graph_objects as go
from optuna.study import StudyDirection
from optuna.trial import TrialState
from optuna.importance import get_param_importances

from .plotting.core import (
    format_hyperparam_name,
    format_title,
    get_plot_filename,
    save_figure,
)
from .plotting.styles import DEFAULT_STYLE
from .plotting.plotly_backend import plot_slice_plotly

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class OptunaDBInfo:
    db_path: Path
    label: str


def discover_optuna_databases(result_dirs: Iterable[Path]) -> List[OptunaDBInfo]:
    """Locate Optuna SQLite databases under given result directories."""
    infos: dict[Path, OptunaDBInfo] = {}
    for base_dir in result_dirs:
        base_dir = Path(base_dir)
        if not base_dir.exists():
            continue

        # Common locations: root or immediate subdirectories.
        candidates = list(base_dir.glob("optuna_study.db"))
        candidates += list(base_dir.glob("**/optuna_study.db"))

        for db_path in candidates:
            if not db_path.is_file():
                continue
            resolved = db_path.resolve()
            if resolved in infos:
                continue
            try:
                relative_label = db_path.parent.relative_to(base_dir).as_posix()
            except ValueError:
                relative_label = db_path.parent.name
            label = relative_label or base_dir.name
            infos[resolved] = OptunaDBInfo(db_path=resolved, label=label)

    sorted_infos = sorted(infos.values(), key=lambda info: info.db_path.as_posix())
    if sorted_infos:
        logger.info("Discovered %d Optuna database(s):", len(sorted_infos))
        for info in sorted_infos:
            logger.info("  - %s (%s)", info.db_path.name, info.db_path)
    return sorted_infos


def generate_optuna_reports(optuna_infos: List[OptunaDBInfo], output_dir: Path) -> None:
    """Generate custom Optuna plots (Plotly + consistent styling) and store them under output/optuna."""
    if not optuna_infos:
        return

    dest_root = (Path(output_dir) / "optuna").resolve()
    if dest_root.exists():
        shutil.rmtree(dest_root)
    dest_root.mkdir(parents=True, exist_ok=True)

    for info in optuna_infos:
        storage_url = f"sqlite:///{info.db_path}"

        try:
            study_summaries = optuna.study.get_all_study_summaries(storage=storage_url)
        except Exception as exc:
            logger.warning("Failed to read Optuna storage %s: %s", info.db_path, exc)
            continue

        if not study_summaries:
            logger.info("No studies found inside %s", info.db_path)
            continue

        for summary in study_summaries:
            try:
                study = optuna.load_study(study_name=summary.study_name, storage=storage_url)
            except Exception as exc:
                logger.warning(
                    "Unable to load study '%s' from %s: %s",
                    summary.study_name,
                    info.db_path,
                    exc,
                )
                continue

            _render_study(
                study=study,
                summary=summary,
                dest_dir=dest_root / info.label / summary.study_name,
                env_label=info.label,
            )


def _render_study(
    study: optuna.Study,
    summary: optuna.study.StudySummary,
    dest_dir: Path,
    env_label: str,
) -> None:
    dest_dir.mkdir(parents=True, exist_ok=True)

    raw_objective_names = (
        summary.user_attrs.get("objective_names")
        or study.user_attrs.get("objective_names")
        or []
    )
    objective_names: list[str] = []
    for idx in range(len(summary.directions)):
        if idx < len(raw_objective_names) and raw_objective_names[idx]:
            objective_names.append(raw_objective_names[idx])
        else:
            objective_names.append(f"objective_{idx + 1}")
    objective_names = [
        format_hyperparam_name(name) if name else f"objective_{idx + 1}"
        for idx, name in enumerate(objective_names)
    ]

    completed_trials = [t for t in study.trials if t.state == TrialState.COMPLETE]
    if not completed_trials:
        logger.info("Study '%s' has no completed trials; skipping plots.", study.study_name)
        return

    trials_df = _build_trials_dataframe(completed_trials, objective_names)
    if trials_df.empty:
        logger.info("Study '%s' trials dataframe is empty after filtering.", study.study_name)
        return

    _plot_optimization_history(
        df=trials_df,
        objective_names=objective_names,
        directions=summary.directions,
        env_label=env_label,
        study_name=study.study_name,
        output_dir=dest_dir,
    )

    _plot_param_importance(
        study=study,
        objective_names=objective_names,
        env_label=env_label,
        output_dir=dest_dir,
    )

    _plot_param_slices(
        trials_df=trials_df,
        objective_names=objective_names,
        env_label=env_label,
        study_name=study.study_name,
        output_dir=dest_dir,
    )


def _build_trials_dataframe(
    trials: list[optuna.trial.FrozenTrial],
    objective_names: list[str],
) -> pd.DataFrame:
    records = []
    for trial in trials:
        record: dict[str, object] = {
            "trial_number": trial.number,
        }

        if trial.values is not None:
            for idx, name in enumerate(objective_names):
                record[name] = trial.values[idx]
        elif trial.value is not None:
            record[objective_names[0]] = trial.value

        if trial.duration is not None:
            record["duration_seconds"] = trial.duration.total_seconds()

        records.append(record | {"params": trial.params})

    df = pd.DataFrame(records)
    if "trial_number" in df.columns:
        df = df.sort_values("trial_number").reset_index(drop=True)
    return df


def _plot_optimization_history(
    df: pd.DataFrame,
    objective_names: list[str],
    directions: tuple[StudyDirection, ...],
    env_label: str,
    study_name: str,
    output_dir: Path,
) -> None:
    if "trial_number" not in df:
        return

    fig = go.Figure()
    palette = DEFAULT_STYLE

    for idx, obj_name in enumerate(objective_names):
        if obj_name not in df:
            continue
        values = df[obj_name].to_numpy(dtype=float)
        direction = directions[idx]

        best_so_far = []
        running = None
        comparator = max if direction == StudyDirection.MAXIMIZE else min
        for val in values:
            if running is None:
                running = val
            else:
                running = comparator(running, val)
            best_so_far.append(running)

        color = palette.color_for(f"objective{idx}", None)
        fig.add_trace(
            go.Scatter(
                x=df["trial_number"],
                y=values,
                mode="markers",
                marker=dict(color=color, size=9, symbol="circle-open"),
                name=f"{obj_name} (trial value)",
            )
        )
        fig.add_trace(
            go.Scatter(
                x=df["trial_number"],
                y=best_so_far,
                mode="lines",
                line=dict(color=color, width=2.5),
                name=f"{obj_name} (best so far)",
            )
        )

    fig.update_layout(
        title=format_title(
            "Optuna Optimization History",
            f"Study: {study_name}",
            backend="plotly",
        ),
        xaxis_title="Trial Number",
        yaxis_title="Objective Value",
        legend=dict(orientation="h", x=0.5, xanchor="center", y=-0.18, yanchor="top"),
        margin=dict(l=60, r=20, t=90, b=140),
        height=520,
    )

    filename = get_plot_filename("optuna_history", env_label, algo=study_name)
    save_figure(fig, output_dir, filename)


def _plot_param_importance(
    study: optuna.Study,
    objective_names: list[str],
    env_label: str,
    output_dir: Path,
) -> None:
    for idx, obj_name in enumerate(objective_names):
        try:
            if len(objective_names) == 1:
                importances = get_param_importances(study)
            else:
                importances = get_param_importances(
                    study,
                    target=lambda t: t.values[idx] if t.values is not None else None,
                    target_name=obj_name,
                )
        except Exception as exc:
            logger.warning("Failed to compute importances for %s: %s", obj_name, exc)
            continue

        if not importances:
            continue

        names = []
        scores = []
        for param, score in importances.items():
            names.append(format_hyperparam_name(param))
            scores.append(float(score))

        fig = go.Figure(
            go.Bar(
                x=scores,
                y=names,
                orientation="h",
                marker=dict(
                    color=[DEFAULT_STYLE.color_for(f"param{i}", None) for i in range(len(scores))]
                ),
            )
        )
        fig.update_layout(
            title=format_title(
                "Optuna Parameter Importances",
                f"Objective: {obj_name}",
                backend="plotly",
            ),
            xaxis_title="Importance",
            yaxis_title="Hyperparameter",
            margin=dict(l=120, r=30, t=80, b=60),
            height=80 * max(3, len(names)),
        )

        filename = get_plot_filename(
            "optuna_param_importance",
            env_label,
            algo=obj_name,
            suffix=study.study_name,
        )
        save_figure(fig, output_dir, filename)


def _plot_param_slices(
    trials_df: pd.DataFrame,
    objective_names: list[str],
    env_label: str,
    study_name: str,
    output_dir: Path,
) -> None:
    params_series = trials_df.get("params")
    if params_series is None:
        return

    params_df = pd.json_normalize(params_series)
    if params_df.empty:
        return

    merged = pd.concat([trials_df.drop(columns=["params"]), params_df], axis=1)

    for idx, obj_name in enumerate(objective_names):
        if obj_name not in merged:
            continue

        tidy_rows = []
        for _, row in merged.iterrows():
            metric = row[obj_name]
            trial_num = row["trial_number"]
            for param_name in params_df.columns:
                raw_value = row.get(param_name)
                numeric_value = _to_float(raw_value)
                if numeric_value is None or not math.isfinite(numeric_value):
                    continue
                tidy_rows.append(
                    {
                        "hp_id": trial_num,
                        "param": param_name,
                        "value": numeric_value,
                        obj_name: metric,
                    }
                )

        if not tidy_rows:
            continue

        tidy_df = pd.DataFrame(tidy_rows)
        tidy_df["param"] = tidy_df["param"].map(format_hyperparam_name)

        algo_label = f"{study_name} · {obj_name}"
        plot_slice_plotly(
            env=f"optuna_{env_label}",
            algo_slice_dict={algo_label: tidy_df},
            output_dir=output_dir,
            metric_column=obj_name,
        )


def _to_float(value) -> float | None:
    if value is None:
        return None
    if isinstance(value, (int, float, np.number)):
        return float(value)
    if isinstance(value, (list, tuple)) and len(value) == 1:
        return _to_float(value[0])
    try:
        return float(value)
    except (TypeError, ValueError):
        return None
