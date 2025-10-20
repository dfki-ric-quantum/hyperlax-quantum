"""Main analysis orchestration."""

import json
import logging
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import List

import pandas as pd
import numpy as np

from .aggregation import aggregate_top_n_across_hps, compute_hp_ranking_metrics
from .data_loader import ExperimentData, load_all_experiments
from .export import export_summaries_to_csv, generate_html_index, _prettify_column_name
from .metrics import compute_all_summaries
from .optuna_integration import (discover_optuna_databases,
                                 generate_optuna_reports)
from .plotting.core import format_hyperparam_name
from .plotting.matplotlib_backend import (plot_aggregate_iqm_mpl,
                                          plot_distribution_combined_mpl,
                                          plot_distribution_mpl,
                                          plot_slice_mpl,
                                          plot_top1_comparison_mpl)
from .plotting.plotly_backend import (plot_aggregate_iqm_plotly,
                                      plot_distribution_plotly,
                                      plot_slice_plotly,
                                      plot_top1_comparison_plotly)
from .runtime_metadata import collect_timing_records


logger = logging.getLogger(__name__)


def _load_json_if_exists(path: Path | None) -> dict:
    if path is None or not path.exists():
        return {}
    try:
        with open(path) as f:
            return json.load(f)
    except Exception as exc:
        logger.warning("Failed to load JSON '%s': %s", path, exc)
        return {}


def _derive_execution_mode(run_args: dict | None, default: str = "unknown") -> str:
    if not isinstance(run_args, dict):
        return default
    if run_args.get("sequential"):
        return "sequential"
    if run_args.get("group_by_structural_hparams"):
        return "grouped_batched"
    batch_size = run_args.get("hparam_batch_size")
    if batch_size == 1:
        return "batched_single"
    if batch_size:
        return "batched"
    return default


def _format_duration(seconds: float | None) -> str:
    if seconds is None:
        return "N/A"
    if isinstance(seconds, (np.ndarray, list, tuple)):
        return "N/A"
    if isinstance(seconds, np.generic):
        seconds = float(seconds)
    elif not isinstance(seconds, (int, float)):
        return "N/A"
    if not np.isfinite(seconds):
        return "N/A"
    seconds = float(seconds)
    if seconds < 0:
        seconds = 0.0
    total_seconds = int(round(seconds))
    days, rem = divmod(total_seconds, 86400)
    hours, rem = divmod(rem, 3600)
    minutes, secs = divmod(rem, 60)
    return f"{days}-{hours:02}:{minutes:02}:{secs:02}"


def _calculate_milestones(total_timesteps: int, num_evals: int) -> list[int]:
    """Calculate evaluation milestones."""
    if num_evals <= 0:
        return []
    return [int(total_timesteps * (i + 1) / num_evals) for i in range(num_evals)]


def _prepare_slice_data(
    hyperparams_df: pd.DataFrame,
    metric_df: pd.DataFrame,
    metric_col: str,
) -> pd.DataFrame:
    """
    Combine hyperparameter values with metric values for slice plots.
    Returns tidy dataframe with columns: hp_id, param, value, raw_value, metric_col.
    """
    if hyperparams_df is None or hyperparams_df.empty:
        return pd.DataFrame()

    if metric_df is None or metric_df.empty or metric_col not in metric_df:
        return pd.DataFrame()

    merged = hyperparams_df.merge(
        metric_df[["hp_id", metric_col]],
        on="hp_id",
        how="inner",
    )
    if merged.empty:
        return pd.DataFrame()

    value_columns = [c for c in merged.columns if c not in {"hp_id", metric_col}]
    if not value_columns:
        return pd.DataFrame()

    tidy = merged.melt(
        id_vars=["hp_id", metric_col],
        value_vars=value_columns,
        var_name="param",
        value_name="raw_value",
    )
    if tidy.empty:
        return tidy

    tidy["value"] = pd.to_numeric(tidy["raw_value"], errors="coerce")
    tidy = tidy.dropna(subset=["value", metric_col])
    if tidy.empty:
        return pd.DataFrame()

    # Filter out parameters with fewer than 2 distinct numeric values
    valid_params = tidy.groupby("param")["value"].nunique()
    valid_params = valid_params[valid_params >= 2].index
    tidy = tidy[tidy["param"].isin(valid_params)]
    if tidy.empty:
        return pd.DataFrame()

    tidy = tidy.sort_values(["param", metric_col, "value"]).reset_index(drop=True)
    return tidy


def run_analysis(
    result_dirs: List[Path],
    output_dir: Path,
    include_algo_variants: bool = True,
    *,
    top_n: int = 5,
    top_n_metric: str = "peak",  # "peak" or "final"
    slice_metric: str = "peak_return",
    max_workers: int | None = None,
    benchmark_root: Path | None = None,
    plot_config: dict | None = None,
    combined_only: bool = False,
):
    """
    End-to-end analysis pipeline.

    Args:
        result_dirs: List of directories containing experiment outputs.
        output_dir: Where plots and CSVs will be written.
        include_algo_variants: Whether to keep variant info (e.g., "(seed42)") in algo labels.
        top_n: Number of best hyperparameter configs to include in aggregate IQM plots.
        top_n_metric: Ranking metric for Top-N selection:
          - "peak": max(mean) over timesteps (classic best-run selection)
          - "final": average mean over the last 10% timesteps
        slice_metric: Metric column to visualize in slice plots
          (e.g., 'peak_return', 'final_return', 'score', 'stability').
        max_workers: Optional thread pool size; set >1 to parallelize per-environment analysis.
        benchmark_root: Optional path to the benchmark run directory (used for metadata export).
        plot_config: Optional dictionary of plotting CLI options for metadata export.
        combined_only: If True, only generate combined distribution plot and skip other analysis.
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    benchmark_root = Path(benchmark_root) if benchmark_root else None
    plot_config = dict(plot_config or {})
    benchmark_args = _load_json_if_exists(benchmark_root / "args.json") if benchmark_root else {}

    benchmark_name = benchmark_args.get("benchmark_name")
    if not benchmark_name and benchmark_root is not None:
        benchmark_name = benchmark_root.name

    benchmark_context = {
        "benchmark_name": benchmark_name,
        "benchmark_root": str(benchmark_root.resolve()) if benchmark_root else "",
        "benchmark_args_path": str(benchmark_root / "args.json") if benchmark_root else "",
        "benchmark_output_root": benchmark_args.get("output_root"),
        "benchmark_run_length_modifier": benchmark_args.get("run_length_modifier"),
        "benchmark_seed": benchmark_args.get("seed"),
        "benchmark_algorithms": benchmark_args.get("algos"),
        "benchmark_environments": benchmark_args.get("envs"),
        "benchmark_sweep_modes": benchmark_args.get("sweep_modes"),
        "benchmark_num_samples_per_run": benchmark_args.get("num_samples_per_run"),
        "benchmark_hparam_batch_size": benchmark_args.get("hparam_batch_size"),
        "benchmark_group_by_structural_hparams": benchmark_args.get("group_by_structural_hparams"),
    }

    optuna_sources = discover_optuna_databases(result_dirs)

    logger.info("Loading experiment data...")
    grouped_data = load_all_experiments(result_dirs, include_algo_variants)

    if not grouped_data:
        logger.error("No valid experiments found!")
        return

    logger.info("\nLoaded experiments:")
    for env, algos in grouped_data.items():
        for algo, exp_data in algos.items():
            n_hps = exp_data.stats_df["hp_id"].nunique()
            logger.info(f"  {env}/{algo}: {n_hps} hyperparameter configs")

    plot_dir = output_dir / "plots"
    plot_dir.mkdir(parents=True, exist_ok=True)

    # If combined_only mode, generate only the combined distribution plot
    if combined_only:
        logger.info("\nGenerating combined distribution plot only...")
        combined_stats_data = {}
        for env, algos_dict in grouped_data.items():
            combined_stats_data[env] = {algo: exp_data.stats_df for algo, exp_data in algos_dict.items()}
        plot_distribution_combined_mpl(combined_stats_data, plot_dir)
        logger.info(f"\nCombined plot complete! Results in: {output_dir.resolve()}")
        return

    # Full analysis mode - continue with all steps
    data_dir = output_dir / "data"
    data_dir.mkdir(parents=True, exist_ok=True)

    combined_metadata_rows: list[dict] = []
    env_items = list(grouped_data.items())

    def _process_environment(env: str, algos_dict: dict[str, ExperimentData]):
        nonlocal combined_metadata_rows
        logger.info(f"\nProcessing environment: {env}")

        example_exp = next(iter(algos_dict.values()))
        metadata = example_exp.metadata or {}
        total_timesteps = int(metadata.get("total_timesteps", 1_000_000))
        num_evals = int(metadata.get("num_evaluation", 20))
        milestones = _calculate_milestones(total_timesteps, num_evals)

        algo_stats_dict: dict[str, pd.DataFrame] = {}
        algo_agg_topn: dict[str, tuple] = {}
        algo_summaries_map: dict[str, list] = {}
        algo_slice_dict: dict[str, pd.DataFrame] = {}
        algo_expdata_map: dict[str, ExperimentData] = {}

        for algo, exp_data in algos_dict.items():
            algo_expdata_map[algo] = exp_data
            stats_df = exp_data.stats_df
            algo_stats_dict[algo] = stats_df

            t_k, iqm_k, lo_k, hi_k = aggregate_top_n_across_hps(
                stats_df, milestones, top_n=top_n, metric=top_n_metric
            )
            algo_agg_topn[algo] = (t_k, iqm_k, lo_k, hi_k)

            summaries = compute_all_summaries(
                exp_data.stats_df,
                budget=total_timesteps,
            )
            algo_summaries_map[algo] = summaries

            summaries_df = (
                pd.DataFrame([s.to_dict() for s in summaries]) if summaries else pd.DataFrame()
            )

            metric_source_df = pd.DataFrame()
            if not summaries_df.empty and slice_metric in summaries_df.columns:
                metric_source_df = summaries_df[["hp_id", slice_metric]].copy()
            else:
                ranking_df = compute_hp_ranking_metrics(
                    exp_data.stats_df,
                    metric=top_n_metric,
                )
                if not ranking_df.empty and slice_metric in ranking_df.columns:
                    metric_source_df = ranking_df[["hp_id", slice_metric]].copy()

            slice_df = _prepare_slice_data(
                exp_data.hyperparams_df,
                metric_source_df,
                metric_col=slice_metric,
            )
            if not slice_df.empty:
                algo_slice_dict[algo] = slice_df

        logger.info("  Generating plots...")

        plot_distribution_mpl(env, algo_stats_dict, plot_dir)
        plot_distribution_plotly(env, algo_stats_dict, plot_dir)

        plot_top1_comparison_mpl(env, algo_stats_dict, plot_dir)
        plot_top1_comparison_plotly(env, algo_stats_dict, plot_dir)

        plot_aggregate_iqm_mpl(env, algo_agg_topn, plot_dir, top_n=top_n)
        plot_aggregate_iqm_plotly(env, algo_agg_topn, plot_dir, top_n=top_n)

        if algo_slice_dict:
            plot_slice_mpl(env, algo_slice_dict, plot_dir, metric_column=slice_metric)
            plot_slice_plotly(env, algo_slice_dict, plot_dir, metric_column=slice_metric)

        logger.info("  Exporting summaries...")
        for algo, summaries in algo_summaries_map.items():
            logger.info(f"    Processing {algo} on {env}...")
            csv_path = data_dir / f"{env}_{algo}_summary.csv"

            exp_data_obj = algo_expdata_map.get(algo)
            run_args = exp_data_obj.run_args if exp_data_obj is not None else {}
            configured_mode = _derive_execution_mode(run_args)
            configured_run_length = run_args.get(
                "run_length_modifier",
                benchmark_context.get("benchmark_run_length_modifier"),
            )
            configured_hbatch = run_args.get("hparam_batch_size")
            configured_num_samples = run_args.get("num_samples")
            configured_sampling = run_args.get("sampling_method")

            hp_id_to_params: dict = {}
            try:
                if exp_data_obj is None or exp_data_obj.hyperparams_df is None:
                    raise KeyError("experiment data not available")

                hyperparams_df = exp_data_obj.hyperparams_df
                raw_param_col = next((col for col in hyperparams_df.columns if col == "hyperparam"), None)
                if raw_param_col:
                    expanded = hyperparams_df.copy()
                    expanded = expanded.drop(columns=[raw_param_col]).join(
                        hyperparams_df[raw_param_col].apply(pd.Series)
                    )
                else:
                    expanded = hyperparams_df.copy()

                if "hp_id" in expanded.columns:
                    expanded = expanded.set_index("hp_id")

                    hp_id_to_params = {}
                    for hp_id, row in expanded.iterrows():
                        params = {}
                        for key, value in row.items():
                            params[format_hyperparam_name(str(key))] = value
                        hp_id_to_params[hp_id] = params
            except Exception as exc:
                logger.warning("Failed to enrich summary with hyperparams for %s/%s: %s", env, algo, exc)

            export_summaries_to_csv(
                summaries,
                csv_path,
                hyperparam_mapping=hp_id_to_params,
            )

            grouped_metadata = pd.DataFrame()
            timing_record_count = 0
            timing_modes = ""
            if exp_data_obj is not None:
                timing_records = collect_timing_records(exp_data_obj.source_dir)
                timing_record_count = int(timing_records.shape[0])
                if not timing_records.empty:
                    timing_modes = ", ".join(
                        sorted({str(x) for x in timing_records["timing_mode"].dropna().unique()})
                    )
                    grouped_metadata = (
                        timing_records.groupby(["config_id", "run_mode"], dropna=False)[
                            ["total_wall_time", "jit_time", "execution_time", "num_samples"]
                        ]
                        .sum(min_count=1)
                        .reset_index()
                    )

                    # grouped_metadata = grouped_metadata.rename(columns={"num_samples": "timing_num_samples"})
                    grouped_metadata = grouped_metadata.drop(columns=["num_samples"])

                    for base_col, formatted_col in (
                        ("total_wall_time", "total_wall_time_readable"),
                        ("jit_time", "jit_time_readable"),
                        ("execution_time", "execution_time_readable"),
                    ):
                        if base_col in grouped_metadata.columns:
                            grouped_metadata[formatted_col] = grouped_metadata[base_col].apply(_format_duration)
                else:
                    grouped_metadata = pd.DataFrame(
                        [
                            {
                                "config_id": exp_data_obj.source_dir.name,
                                "run_mode": configured_mode,
                                "total_wall_time": None,
                                "jit_time": None,
                                "execution_time": None,
                                #"timing_num_samples": None,
                                "total_wall_time_readable": _format_duration(None),
                                "jit_time_readable": _format_duration(None),
                                "execution_time_readable": _format_duration(None),
                            }
                        ]
                    )
            else:
                grouped_metadata = pd.DataFrame(
                    [
                        {
                            "config_id": "unknown",
                            "run_mode": configured_mode,
                            "total_wall_time": None,
                            "jit_time": None,
                            "execution_time": None,
                            #"timing_num_samples": None,
                            "total_wall_time_readable": _format_duration(None),
                            "jit_time_readable": _format_duration(None),
                            "execution_time_readable": _format_duration(None),
                        }
                    ]
                )

            grouped_metadata["config_id"] = grouped_metadata["config_id"].astype(str)
            grouped_metadata["environment"] = env
            grouped_metadata["algorithm"] = algo
            grouped_metadata["source_dir"] = (
                str(exp_data_obj.source_dir.resolve()) if exp_data_obj is not None else ""
            )
            grouped_metadata["configured_run_mode"] = configured_mode
            grouped_metadata["configured_hparam_batch_size"] = configured_hbatch
            grouped_metadata["configured_num_samples"] = configured_num_samples
            grouped_metadata["configured_sampling_method"] = configured_sampling
            grouped_metadata["configured_group_by_structural_hparams"] = run_args.get(
                "group_by_structural_hparams"
            )
            grouped_metadata["configured_run_length_modifier"] = configured_run_length
            grouped_metadata["timing_record_count"] = timing_record_count
            grouped_metadata["timing_modes"] = timing_modes
            grouped_metadata["benchmark_name"] = benchmark_context.get("benchmark_name")
            grouped_metadata["benchmark_run_length_modifier"] = benchmark_context.get(
                "benchmark_run_length_modifier"
            )
            grouped_metadata["benchmark_seed"] = benchmark_context.get("benchmark_seed")

            combined_metadata_rows.extend(grouped_metadata.to_dict("records"))


    worker_count = max_workers if max_workers and max_workers > 1 else 1
    if worker_count > 1 and len(env_items) > 1:
        try:
            import matplotlib.pyplot as plt  # noqa: WPS433 (runtime import to switch backend)

            current_backend = plt.get_backend()
            if current_backend.lower() != "agg":
                logger.info(
                    "Switching Matplotlib backend to 'Agg' for thread-safe plotting (was %s)",
                    current_backend,
                )
                plt.switch_backend("Agg")
        except Exception as exc:  # pragma: no cover - best effort
            logger.warning("Could not switch Matplotlib backend for multithreaded analysis: %s", exc)

        logger.info(
            "Running analysis across %d environments using %d worker threads",
            len(env_items),
            worker_count,
        )
        with ThreadPoolExecutor(max_workers=worker_count) as executor:
            future_map = {
                executor.submit(_process_environment, env, algos_dict): env
                for env, algos_dict in env_items
            }
            for future in as_completed(future_map):
                env = future_map[future]
                try:
                    future.result()
                except Exception as exc:  # pragma: no cover - log unexpected errors
                    logger.error("Environment %s analysis failed: %s", env, exc, exc_info=True)
    else:
        for env, algos_dict in env_items:
            _process_environment(env, algos_dict)

    # Generate combined distribution plot after all environments are processed
    logger.info("\nGenerating combined distribution plot...")
    combined_stats_data = {}
    for env, algos_dict in grouped_data.items():
        combined_stats_data[env] = {algo: exp_data.stats_df for algo, exp_data in algos_dict.items()}
    plot_distribution_combined_mpl(combined_stats_data, plot_dir)

    if combined_metadata_rows:
        combined_df = pd.DataFrame(combined_metadata_rows)
        # Ensure config_id is the leading column for easier comparisons
        cols = ["config_id"] + [c for c in combined_df.columns if c != "config_id"]
        combined_df = combined_df[cols]

        combined_df = combined_df.rename(columns=_prettify_column_name)

        combined_csv = data_dir / "summary_metadata.csv"
        combined_df.to_csv(combined_csv, index=False)

    # HTML index
    if optuna_sources:
        generate_optuna_reports(optuna_sources, output_dir)

    logger.info("\nGenerating HTML index...")
    generate_html_index(output_dir)

    logger.info(f"\nAnalysis complete! Results in: {output_dir.resolve()}")
