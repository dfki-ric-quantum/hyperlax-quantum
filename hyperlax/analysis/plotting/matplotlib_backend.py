"""Matplotlib plotting functions."""

import re
from pathlib import Path

import matplotlib.pyplot as plt
from matplotlib.cm import ScalarMappable
from matplotlib.colors import Normalize
import numpy as np
import pandas as pd
import seaborn as sns

from .core import (format_algo_name, format_env_name, format_hyperparam_name,
                   format_title, get_plot_filename,
                   present_families_fas_from_labels, save_figure,
                   split_family_fa)
from .styles import DEFAULT_STYLE, FA_ORDER, FAMILY_ORDER
from ..aggregation import select_top_n_hp_ids

def plot_distribution_mpl(
    env: str,
    algo_data_dict: dict[str, pd.DataFrame],
    output_dir: Path,
):
    """
    Distribution of best-found returns across hyperparameters (family/FA + variant aware).
    Expects each stats_df to have columns: ['hp_id','mean', ...]
    """
    rows = []
    labels = []
    for algo, stats_df in algo_data_dict.items():
        peak_returns = stats_df.groupby('hp_id')['mean'].max()
        for _, peak in peak_returns.items():
            rows.append({'label': algo, 'peak_return': peak})
        labels.append(algo)

    if not rows:
        return

    df = pd.DataFrame(rows)

    # Present families/FAs (ordered)
    families, fas = present_families_fas_from_labels(labels, FAMILY_ORDER, FA_ORDER)

    # Parse into components (family, fa, variant)
    parsed = df["label"].map(split_family_fa)
    df["family"]  = [p[0] for p in parsed]
    df["fa"]      = [p[1] for p in parsed]
    df["variant"] = [p[2] for p in parsed]

    # Canonical label (category) including variant when present
    def make_canon(row):
        base = f"{row['family']}-{row['fa']}" if row['fa'] else row['family']
        return f"{base} ({row['variant']})" if row['variant'] else base

    df["canon"] = df.apply(make_canon, axis=1)

    # Build ordered categories: families → FAs (if any) → variants (lexical)
    ordered_labels: list[str] = []
    if fas:
        for fam in families:
            for fa in fas:
                subset = df[(df["family"] == fam) & (df["fa"] == fa)]
                if subset.empty:
                    continue
                variants = sorted(v for v in subset["variant"].unique() if v)
                base = f"{fam}-{fa}"
                if variants:
                    ordered_labels.extend([f"{base} ({v})" for v in variants])
                else:
                    ordered_labels.append(base)
        # Base palette per family/FA (variants share the same base color)
        palette = {key: DEFAULT_STYLE.color_for(*key.split('-', 1))
                   for key in [f"{fam}-{fa}" for fam in families for fa in fas]}
    else:
        for fam in families:
            subset = df[df["family"] == fam]
            if subset.empty:
                continue
            variants = sorted(v for v in subset["variant"].unique() if v)
            base = fam
            if variants:
                ordered_labels.extend([f"{base} ({v})" for v in variants])
            else:
                ordered_labels.append(base)
        palette = {fam: DEFAULT_STYLE.color_for(fam, None) for fam in families}

    # Map each canon category to its base (family or family-FA) color
    def _base_of(canon_label: str) -> str:
        return canon_label.split(' (', 1)[0] if ' (' in canon_label else canon_label

    canon_palette = {canon: palette.get(_base_of(canon), "#333333")
                     for canon in ordered_labels}

    # Dynamic width like before
    fig_w = max(12, 0.5 * len(ordered_labels) + 6)
    fig, ax = plt.subplots(figsize=(fig_w, 8), dpi=150)

    # Use hue==x (canon) to color categories without the deprecation warning
    sns.boxplot(
        data=df,
        x="canon", y="peak_return",
        order=ordered_labels,
        hue="canon",
        palette=canon_palette,
        dodge=False,
        ax=ax,
        saturation=1, linewidth=1.0, fliersize=0
    )

    # Hide the redundant legend if Seaborn added one
    leg = ax.get_legend()
    if leg is not None:
        leg.remove()

    # Overlay jittered points with FA-specific markers in black
    rng = np.random.default_rng(42)
    x_positions = {lbl: i for i, lbl in enumerate(ordered_labels)}
    for lbl in ordered_labels:
        sub = df[df["canon"] == lbl]["peak_return"].to_numpy()
        if sub.size == 0:
            continue
        xs = x_positions[lbl] + rng.uniform(-0.18, 0.18, size=sub.size)
        # Determine FA of this label (if any)
        base = _base_of(lbl)
        fa = base.split("-", 1)[1] if "-" in base else None
        ax.scatter(xs, sub, s=36,
                   marker=DEFAULT_STYLE.mpl_marker_for_fa(fa),
                   color="black", alpha=0.85, zorder=3)

    title = "Distribution of Best-Found Mean Returns across HP Trials"
    subtitle = f"on {format_env_name(env)}"
    ax.set_title(format_title(title, subtitle, "matplotlib"))
    ax.set_xlabel("")
    ax.set_ylabel("Best-Found Mean Returns")

    # Explicitly set tick positions before tick labels to avoid the UserWarning.
    ax.set_xticks(list(range(len(ordered_labels))))
    ax.set_xticklabels(ordered_labels, rotation=45, ha="right")

    ax.grid(True, axis="y", linestyle="--", alpha=0.35)

    plt.tight_layout()
    filename = get_plot_filename("best_found_return_distribution_across_hyperparams", env)
    save_figure(fig, output_dir, filename)


def slugify_for_path(text: str) -> str:
    # Clean, filename-safe slug (no spaces, parentheses)
    text = text.strip().lower()
    # Remove parentheses entirely
    text = re.sub(r"[()]", "", text)
    # Remove any characters that aren't word chars, dash, underscore, dot, or brackets
    text = re.sub(r"[^\w\-\.\[\]]+", "_", text)
    # Collapse multiple underscores
    text = re.sub(r"_+", "_", text)
    return text[:120]

def _save_individual_figure(fig, output_dir: Path, subdir: str, filename_stem: str):
    """
    Save `fig` to output_dir/plots/<subdir>/<filename_stem>.png
    (doesn't rely on project-specific save_figure)
    """
    target_dir = output_dir / subdir
    target_dir.mkdir(parents=True, exist_ok=True)
    out_path = target_dir / f"{filename_stem}.png"
    fig.savefig(out_path, bbox_inches="tight", dpi=150)
    plt.close(fig)


def plot_top1_comparison_mpl(
    env: str,
    algo_data_dict: dict[str, pd.DataFrame],
    output_dir: Path,
    selection_metric: str = "peak",
    tail_frac: float = 0.10,
    tol: float = 1e-6,
):
    """
    Top-1 run comparison with CI bands, family/FA markers, unified legend (variants shown separately).
    """
    fig, ax = plt.subplots(figsize=(12, 8), dpi=150)
    style = DEFAULT_STYLE

    for algo, stats_df in algo_data_dict.items():
        top_ids = select_top_n_hp_ids(
            stats_df=stats_df,
            n=1,
            metric=selection_metric,
            tail_frac=tail_frac,
            tol=tol,
        )
        if not top_ids:
            # nothing to plot for this algo
            continue

        top_hp = top_ids[0]

        top_data = stats_df[stats_df["hp_id"] == top_hp].sort_values("timestep", kind="mergesort")
        if top_data.empty:
            continue

        fam, fa, _variant = split_family_fa(algo)
        color  = style.color_for(fam, fa)
        marker = style.mpl_marker_for_fa(fa)

        label = format_algo_name(algo)
        if "sample_id" in top_data.columns and not top_data["sample_id"].empty:
            sid = str(top_data["sample_id"].iloc[0])
            label = f"{label} (ID: {sid})"

        ax.plot(
            top_data["timestep"],
            top_data["mean"],
            color=color,
            marker=marker,
            label=label,
            linewidth=2.5,
            markersize=6,
        )
        if {"std"}.issubset(top_data.columns):
            ax.fill_between(
                top_data["timestep"],
                top_data["mean"] - top_data["std"],
                top_data["mean"] + top_data["std"],
                color=color,
                alpha=0.2,
            )

        fig_i, ax_i = plt.subplots(figsize=(10, 7), dpi=150)
        ax_i.plot(
            top_data["timestep"],
            top_data["mean"],
            color=color,
            marker=marker,
            linewidth=2.5,
            markersize=6,
        )
        if {"std"}.issubset(top_data.columns):
            ax_i.fill_between(
                top_data["timestep"],
                top_data["mean"] - top_data["std"],
                top_data["mean"] + top_data["std"],
                color=color,
                alpha=0.2,
            )
        ind_title = f"Top-1 Run - {format_algo_name(algo)}"
        sel_txt   = f"top-1 by {selection_metric} (hybrid tie-break)"
        ind_sub   = f"on {format_env_name(env)} - {sel_txt}"
        ax_i.set_title(format_title(ind_title, ind_sub, "matplotlib"), pad=10)
        ax_i.set_xlabel("Timesteps")
        ax_i.set_ylabel("Mean Return (± Std Dev)")
        ax_i.grid(True)

        env_slug  = slugify_for_path(format_env_name(env))
        algo_slug = slugify_for_path(format_algo_name(algo))
        filename_stem = f"top1_{env_slug}__{algo_slug}"
        _save_individual_figure(fig_i, output_dir, subdir="top-1", filename_stem=filename_stem)

    title = "Performance of Top-1 Hyperparameter Run"
    subtitle = f"on {format_env_name(env)} - top-1 by {selection_metric}; tie-break (score ↓ → earliest ↑ → stability ↓)"
    ax.set_title(format_title(title, subtitle, "matplotlib"), pad=10)
    ax.set_xlabel("Timesteps")
    ax.set_ylabel("Mean Return (± Std Dev)")
    ax.legend()
    plt.grid(True)

    filename = get_plot_filename("top1_run_comparison", env)
    save_figure(fig, output_dir, filename)


def plot_aggregate_iqm_mpl(
    env: str,
    algo_agg_dict: dict[str, tuple],
    output_dir: Path,
    top_n: int = 5,
):
    """
    Aggregate IQM curves with bootstrap CIs (variants shown separately).
    Also saves one figure per algorithm under /top-<n>/.
    algo_agg_dict: {algorithm: (timesteps, iqm, ci_lower, ci_upper)}
    """
    fig, ax = plt.subplots(figsize=(12, 8), dpi=150)

    style = DEFAULT_STYLE
    per_dir = f"top-{top_n}" if top_n is not None else "top-n"

    for algo, (timesteps, iqm, ci_low, ci_high) in algo_agg_dict.items():
        fam, fa, _variant = split_family_fa(algo)
        color = style.color_for(fam, fa)
        marker = style.mpl_marker_for_fa(fa)

        ax.plot(
            timesteps,
            iqm,
            color=color,
            linestyle="-",
            marker=marker,
            markersize=6,
            label=format_algo_name(algo),
            linewidth=2.5,
        )
        ax.fill_between(timesteps, ci_low, ci_high, color=color, alpha=0.2)

        fig_i, ax_i = plt.subplots(figsize=(10, 7), dpi=150)
        ax_i.plot(
            timesteps,
            iqm,
            color=color,
            linestyle="-",
            marker=marker,
            markersize=6,
            linewidth=2.5,
        )
        ax_i.fill_between(timesteps, ci_low, ci_high, color=color, alpha=0.2)
        ind_title = f"Aggregate Performance (IQM) - {format_algo_name(algo)}"
        n_text = f"Top-{top_n}"
        ind_sub = f"on {format_env_name(env)} ({n_text}, 95% CI)"
        ax_i.set_title(format_title(ind_title, ind_sub, "matplotlib"), pad=10)
        ax_i.set_xlabel("Timesteps")
        ax_i.set_ylabel(f"Return (IQM)")
        ax_i.grid(True)

        env_slug = slugify_for_path(format_env_name(env))
        algo_slug = slugify_for_path(format_algo_name(algo))
        n_slug = f"top{top_n}"
        filename_stem = f"{n_slug}_iqm_{env_slug}__{algo_slug}"
        _save_individual_figure(fig_i, output_dir, subdir=per_dir, filename_stem=filename_stem)

    title = f"Aggregate Performance of Top-{top_n} HP Trials"
    subtitle = f"on {format_env_name(env)} (IQM with 95% CI)"
    ax.set_title(format_title(title, subtitle, "matplotlib"), pad=10)
    ax.set_xlabel("Timesteps")
    ax.set_ylabel(f"Return (IQM)")
    ax.legend()
    plt.grid(True)

    filename = get_plot_filename(f"top{top_n}_aggregate_perf", env, suffix="iqm")
    save_figure(fig, output_dir, filename)


def plot_distribution_combined_mpl(
    grouped_data: dict[str, dict[str, pd.DataFrame]],
    output_dir: Path,
):
    """
    Create a combined figure with all env+algo combinations as subplots in a 2D grid.
    Rows represent environments, columns represent algorithms.

    This version uses a robust layout strategy that avoids manual positioning.

    Args:
        grouped_data: {env: {algo: stats_df}}
        output_dir: Output directory for the plot
    """
    if not grouped_data:
        return

    # Get the list of environments
    envs = list(grouped_data.keys())

    # Helper function to sort algorithms using FAMILY_ORDER and FA_ORDER
    def sort_algos(algo_list):
        """Sort algorithms based on FAMILY_ORDER and FA_ORDER."""
        if not algo_list:
            return []

        families, fas = present_families_fas_from_labels(
            algo_list, FAMILY_ORDER, FA_ORDER
        )

        ordered = []
        if fas:
            for fam in families:
                for fa in fas:
                    matching = [
                        algo for algo in algo_list
                        if split_family_fa(algo)[0] == fam and split_family_fa(algo)[1] == fa
                    ]
                    matching.sort(key=lambda a: split_family_fa(a)[2] or "")
                    ordered.extend(matching)
        else:
            for fam in families:
                matching = [
                    algo for algo in algo_list
                    if split_family_fa(algo)[0] == fam and split_family_fa(algo)[1] is None
                ]
                matching.sort(key=lambda a: split_family_fa(a)[2] or "")
                ordered.extend(matching)

        return ordered

    # For each environment, get sorted list of algorithms
    env_algos = {}
    max_algos = 0
    for env in envs:
        algos = list(grouped_data[env].keys())
        sorted_algos = sort_algos(algos)
        env_algos[env] = sorted_algos
        max_algos = max(max_algos, len(sorted_algos))

    nrows = len(envs)
    ncols = max_algos

    # Calculate figure size dynamically
    # Use smaller subplot sizes for better overall layout
    subplot_width = 3.5
    subplot_height = 4.0

    # Add extra space for title and labels
    title_space = 0.8  # inches for main title
    label_space_per_env = 0.4  # inches for each env label

    total_height = (subplot_height * nrows) + title_space + (label_space_per_env * nrows)

    # Create figure WITHOUT constrained layout (causes issues with many subplots)
    fig = plt.figure(
        figsize=(subplot_width * ncols, total_height),
        dpi=150
    )

    # Create grid spec with space for environment labels
    # Add an extra row for each environment to place the label
    gs = fig.add_gridspec(
        nrows=nrows * 2,  # Double rows: one for label, one for plots
        ncols=ncols,
        height_ratios=[0.12, 1] * nrows,  # Smaller row for label, large for plot
        hspace=0.3,
        wspace=0.2,
        top=0.96,
        bottom=0.03,
        left=0.06,
        right=0.99
    )

    # Plot each env (row) and algo (column) combination
    for env_idx, env in enumerate(envs):
        algo_dict = grouped_data[env]
        algos = env_algos[env]

        # Calculate actual row indices (accounting for label rows)
        label_row = env_idx * 2
        plot_row = env_idx * 2 + 1

        # Add environment label spanning all columns in the label row
        ax_label = fig.add_subplot(gs[label_row, :])
        ax_label.text(
            0.5, 0.3,
            format_env_name(env),
            ha='center',
            va='center',
            fontsize=20,
            #fontweight='bold',
            transform=ax_label.transAxes
        )
        ax_label.axis('off')

        # First pass: compute min/max across all algorithms in this environment (row)
        all_peak_returns = []
        for algo in algos:
            stats_df = algo_dict[algo]
            peak_returns = stats_df.groupby('hp_id')['mean'].max()
            all_peak_returns.extend(peak_returns.values)

        if all_peak_returns:
            y_min = min(all_peak_returns)
            y_max = max(all_peak_returns)
            # Add padding (5% on each side)
            y_range = y_max - y_min
            if y_range > 0:
                y_min = y_min - 0.05 * y_range
                y_max = y_max + 0.05 * y_range
            else:
                # Handle case where all values are the same
                y_min = y_min - 1
                y_max = y_max + 1
        else:
            y_min, y_max = 0, 1

        # Second pass: plot each algorithm with consistent y-axis limits
        for col_idx, algo in enumerate(algos):
            ax = fig.add_subplot(gs[plot_row, col_idx])
            stats_df = algo_dict[algo]

            # Compute peak returns per hp_id
            peak_returns = stats_df.groupby('hp_id')['mean'].max()
            rows = [{'peak_return': peak} for peak in peak_returns]

            if not rows:
                ax.set_visible(False)
                continue

            df = pd.DataFrame(rows)

            # Parse algorithm name for styling
            fam, fa, variant = split_family_fa(algo)
            color = DEFAULT_STYLE.color_for(fam, fa)

            # Create boxplot
            sns.boxplot(
                data=df,
                y="peak_return",
                ax=ax,
                color=color,
                saturation=0.9,
                linewidth=1.5,
                fliersize=0,
                width=0.2,
            )
            # Apply transparency to boxplot elements
            for patch in ax.patches:
                patch.set_alpha(0.8)
            for line in ax.lines:
                line.set_alpha(0.9)

            # Overlay jittered points
            rng = np.random.default_rng(42 + env_idx * ncols + col_idx)
            ys = df["peak_return"].to_numpy()
            xs = rng.uniform(-0.08, 0.08, size=ys.size)
            ax.scatter(
                xs, ys,
                s=40,
                marker=DEFAULT_STYLE.mpl_marker_for_fa(fa),
                color="black",
                alpha=0.85,
                zorder=3,
            )

            # Set consistent y-axis limits for this row (environment)
            ax.set_ylim(y_min, y_max)

            # Y-axis label only for first column
            if col_idx == 0:
                ax.set_ylabel("Best-Found Mean Returns", fontsize=16)#, fontweight='bold')
            else:
                ax.set_ylabel("")

            # Place algo name as x-axis label below the plot
            ax.set_xlabel(format_algo_name(algo), fontsize=18)#, fontweight='bold')
            ax.set_xticks([])
            ax.tick_params(axis='y', labelsize=14)

            # Y-axis label only for first column
            if col_idx == 0:
                ax.set_ylabel("Best-Found Mean Returns", fontsize=16) #, fontweight='bold')
            else:
                ax.set_ylabel("")
                ax.set_yticklabels([])  # Hide y-tick labels for non-first columns

            ax.grid(True, axis="y", linestyle='--', alpha=0.4, linewidth=0.8)

            # Remove the bounding box (spines) while keeping grid
            #ax.spines['top'].set_visible(False)
            #ax.spines['right'].set_visible(False)
            #ax.spines['bottom'].set_visible(False)
            #ax.spines['left'].set_visible(False)


        # Hide unused subplots in this row
        for col_idx in range(len(algos), ncols):
            ax = fig.add_subplot(gs[plot_row, col_idx])
            ax.set_visible(False)

    # Overall title using suptitle with proper spacing
    fig.suptitle(
        "Distribution of Best-Found Mean Returns across HP Trials",
        fontsize=24,
        fontweight='bold',
        y=0.995
    )

    filename = "best_found_return_distribution_combined"
    save_figure(fig, output_dir, filename)

def plot_slice_mpl(
    env: str,
    algo_slice_dict: dict[str, pd.DataFrame],
    output_dir: Path,
    *,
    metric_column: str = "peak_return",
):
    """
    Create Optuna-style slice plots (one subplot per hyperparameter) for each algorithm.

    Args:
        env: Environment name.
        algo_slice_dict: {algo: tidy dataframe with columns ['hp_id','param','value',metric_column,...]}.
        output_dir: Destination directory.
        metric_column: Summary metric to plot on the Y axis (default: 'peak_return').
    """
    if not algo_slice_dict:
        return

    metric_pretty = metric_column.replace("_", " ").title()

    for algo, df in algo_slice_dict.items():
        if df is None or df.empty:
            continue

        # Ensure required columns exist
        required = {"param", "value", metric_column}
        if not required.issubset(df.columns):
            continue

        # Drop NaNs that may still be present
        df_clean = df.dropna(subset=["value", metric_column]).copy()
        if df_clean.empty:
            continue

        # Only keep parameters that have variation
        params = [
            param
            for param, group in df_clean.groupby("param")
            if group["value"].nunique(dropna=True) >= 2
        ]
        if not params:
            continue
        params.sort()

        n_params = len(params)
        nrows = 1
        ncols = n_params

        fig_w = max(8.0, 3.5 * ncols)
        fig_h = 4.5
        fig, axes = plt.subplots(
            nrows=nrows,
            ncols=ncols,
            figsize=(fig_w, fig_h),
            dpi=150,
            squeeze=False,
            layout="constrained",
        )
        axes_array = axes

        metric_values = df_clean[metric_column].to_numpy(dtype=float)
        metric_values = metric_values[np.isfinite(metric_values)]
        use_color_scale = metric_values.size > 0 and not np.allclose(
            metric_values.max(), metric_values.min()
        )
        if use_color_scale:
            norm = Normalize(float(metric_values.min()), float(metric_values.max()))
            cmap = plt.get_cmap("viridis")
        else:
            norm = None
            cmap = None

        active_axes: list = []

        for idx, param in enumerate(params):
            row = 0
            col = idx
            ax = axes_array[row, col]
            subset = df_clean[df_clean["param"] == param]
            x = subset["value"].to_numpy(dtype=float)
            y = subset[metric_column].to_numpy(dtype=float)
            mask = np.isfinite(x) & np.isfinite(y)
            x = x[mask]
            y = y[mask]
            if x.size == 0:
                ax.set_visible(False)
                continue

            active_axes.append(ax)

            if use_color_scale and cmap is not None and norm is not None:
                colors = cmap(norm(y))
                ax.scatter(
                    x,
                    y,
                    s=40,
                    c=colors,
                    edgecolor="black",
                    linewidths=0.3,
                    alpha=0.9,
                )
            else:
                ax.scatter(
                    x,
                    y,
                    s=40,
                    color="#1f77b4",
                    edgecolor="black",
                    linewidths=0.3,
                    alpha=0.85,
                )

            # Heuristic for log scale
            finite_x = x[np.isfinite(x)]
            if finite_x.size > 0 and np.all(finite_x > 0):
                x_min = float(finite_x.min())
                x_max = float(finite_x.max())
                if x_min > 0 and (x_max / max(x_min, 1e-12)) >= 100.0:
                    ax.set_xscale("log")

            ax.set_xlabel(format_hyperparam_name(param))
            if col == 0:
                ax.set_ylabel(metric_pretty)
            else:
                ax.set_ylabel("")
            ax.grid(True, linestyle="--", alpha=0.3)

        # Hide unused axes (e.g., padding cells)
        axes_flat = axes_array.ravel()
        for extra_ax in axes_flat[n_params:]:
            extra_ax.set_visible(False)

        if use_color_scale and cmap is not None and norm is not None and active_axes:
            sm = ScalarMappable(norm=norm, cmap=cmap)
            sm.set_array([])
            fig.colorbar(
                sm,
                ax=active_axes,
                pad=0.02,
                fraction=0.035,
                label=metric_pretty,
            )

        title = f"Hyperparameter Slice Plot - {format_algo_name(algo)}"
        subtitle = f"on {format_env_name(env)} - {metric_pretty}"
        fig.suptitle(format_title(title, subtitle, "matplotlib"), y=1.1)

        filename = get_plot_filename(
            "slice_plot",
            env,
            algo=format_algo_name(algo),
            suffix=metric_column,
        )
        save_figure(fig, output_dir, filename)
