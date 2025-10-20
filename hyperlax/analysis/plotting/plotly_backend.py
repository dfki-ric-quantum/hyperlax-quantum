"""Plotly interactive plotting functions."""

from pathlib import Path

import numpy as np
import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from matplotlib.colors import to_rgba

from ..aggregation import select_top_n_hp_ids
from .core import (format_algo_name, format_env_name, format_hyperparam_name,
                   format_title, get_plot_filename,
                   present_families_fas_from_labels, save_figure,
                   split_family_fa)
from .styles import DEFAULT_STYLE, FA_ORDER, FAMILY_ORDER


def _rgba_fill(hex_color: str, alpha: float = 0.2) -> str:
    r, g, b, _ = to_rgba(hex_color)
    return f"rgba({int(r*255)},{int(g*255)},{int(b*255)},{alpha})"


def _reorder_plotly_legend(fig: go.Figure, series_order: list[str]) -> None:
    """Reorder legend groups to match desired order (family/fa/variant keys)."""
    groups = {}
    key_order = []
    for i, t in enumerate(fig.data):
        key = getattr(t, "legendgroup", None) or getattr(t, "name", f"__trace_{i}")
        key_order.append(key)
        groups.setdefault(key, []).append(i)

    ordered_keys = [k for k in series_order if k in groups]
    seen = set(ordered_keys)
    for k in key_order:
        if k not in seen:
            ordered_keys.append(k); seen.add(k)

    new_indices = [idx for k in ordered_keys for idx in groups[k]]
    fig.data = tuple(fig.data[i] for i in new_indices)


def plot_distribution_plotly(
    env: str,
    algo_data_dict: dict[str, pd.DataFrame],
    output_dir: Path,
):
    """Interactive box plot of peak return distributions; FA + variant aware; supports family-only."""
    rows, labels = [], []
    for algo, stats_df in algo_data_dict.items():
        peak_returns = stats_df.groupby('hp_id')['mean'].max()
        for _, peak in peak_returns.items():
            rows.append({'label': algo, 'peak_return': peak})
        labels.append(algo)

    if not rows:
        return

    df = pd.DataFrame(rows)
    families, fas = present_families_fas_from_labels(labels, FAMILY_ORDER, FA_ORDER)

    # Components (family, fa, variant)
    parsed = df["label"].map(split_family_fa)
    df["family"]  = [p[0] for p in parsed]
    df["fa"]      = [p[1] for p in parsed]
    df["variant"] = [p[2] for p in parsed]

    # Canonical key including variant when present
    def make_canon(row):
        base = f"{row['family']}-{row['fa']}" if row['fa'] else row['family']
        return f"{base} ({row['variant']})" if row['variant'] else base

    df["canon"] = df.apply(make_canon, axis=1)

    # Ordered categories
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

    style = DEFAULT_STYLE
    fig = go.Figure()

    # One Box trace per canonical label; color/marker from (family, fa), independent of variant
    for canon in ordered_labels:
        # Recover family/fa from base part
        base = canon.split(' (', 1)[0] if ' (' in canon else canon
        if "-" in base:
            fam, fa = base.split("-", 1)
        else:
            fam, fa = base, None

        color_hex = style.color_for(fam, fa)
        data = df[df["canon"] == canon]["peak_return"]

        fig.add_trace(go.Box(
            x=[canon] * len(data),
            y=data,
            name=canon,
            fillcolor=color_hex,
            line=dict(color="rgba(0,0,0,0.85)", width=1),
            boxmean=False,
            boxpoints="all",
            jitter=0.4,
            pointpos=0.0,
            marker=dict(
                color="black",
                symbol=style.plotly_marker_for_fa(fa),
                size=8,
                line=dict(width=0),
                opacity=0.6
            ),
            showlegend=False,
        ))

    fig.update_layout(
        title=format_title("Distribution of Best-Found Returns across HP Trials",
                           f"on {format_env_name(env)}", backend="plotly"),
        yaxis_title="Best-Found Mean Returns (per run)",
        xaxis_title="",
        xaxis=dict(categoryorder="array", categoryarray=ordered_labels),
        legend=dict(
            orientation="h", x=0.5, xanchor="center",
            y=-0.12, yanchor="top", traceorder="grouped",
            borderwidth=0
        ),
        margin=dict(l=60, r=20, t=90, b=140),
        height=650
    )

    filename = get_plot_filename("best_found_return_distribution_across_hyperparams", env)
    save_figure(fig, output_dir, filename)

def plot_top1_comparison_plotly(
    env: str,
    algo_data_dict: dict[str, pd.DataFrame],
    output_dir: Path,
    selection_metric: str = "peak",   # NEW: "peak" or "final"
    tail_frac: float = 0.10,          # NEW: for "final" & stability tie-break
    tol: float = 1e-6,                # NEW: tolerance for earliest-reach
):
    """Interactive top-1 comparison (hybrid tie-break) with grouped CI bands and ordered legend; variants shown separately."""
    fig = go.Figure()
    style = DEFAULT_STYLE

    present_labels = list(algo_data_dict.keys())
    families, fas = present_families_fas_from_labels(present_labels, FAMILY_ORDER, FA_ORDER)

    # Build desired ordering including variants
    def canon_from_label(lbl: str) -> str:
        fam, fa, var = split_family_fa(lbl)
        base = f"{fam}-{fa}" if fa else fam
        return f"{base} ({var})" if var else base

    all_canons = [canon_from_label(l) for l in present_labels]

    def ordered_canons():
        out = []
        if fas:
            for fam in families:
                for fa in fas:
                    base = f"{fam}-{fa}"
                    group = [c for c in all_canons if c.startswith(base)]
                    bare = [base] if base in group else []
                    vars_ = sorted([c for c in group if c != base])
                    out.extend(bare + vars_)
        else:
            for fam in families:
                base = fam
                group = [c for c in all_canons if c == base or c.startswith(f"{base} (")]
                bare = [base] if base in group else []
                vars_ = sorted([c for c in group if c != base])
                out.extend(bare + vars_)
        return out

    desired_legend_order = ordered_canons()
    groups_present = set()

    for algo, stats_df in algo_data_dict.items():
        # ---- pick top-1 HP with hybrid criteria (replaces plain max()) ----
        top_ids = select_top_n_hp_ids(
            stats_df=stats_df,
            n=1,
            metric=selection_metric,
            tail_frac=tail_frac,
            tol=tol,
        )
        if not top_ids:
            continue
        top_hp = top_ids[0]

        top_data = (
            stats_df[stats_df['hp_id'] == top_hp]
            .sort_values('timestep', kind='mergesort')
        )
        if top_data.empty:
            continue

        fam, fa, _variant = split_family_fa(algo)
        color_hex = style.color_for(fam, fa)
        fill_color = _rgba_fill(color_hex, 0.2)
        marker = style.plotly_marker_for_fa(fa)

        canon = canon_from_label(algo)  # unique per variant
        groups_present.add(canon)

        # Optional sample id
        name = format_algo_name(algo)
        if "sample_id" in top_data.columns and not top_data["sample_id"].empty:
            name = f"{name} (ID: {top_data['sample_id'].iloc[0]})"

        x = top_data['timestep'].to_numpy()
        y_mean = top_data['mean'].to_numpy()

        # CI band only if std is available & finite
        has_std = "std" in top_data.columns and top_data["std"].notna().any()
        if has_std:
            y_std = top_data['std'].to_numpy()
            fig.add_trace(go.Scatter(
                x=np.concatenate([x, x[::-1]]),
                y=np.concatenate([y_mean + y_std, (y_mean - y_std)[::-1]]),
                fill='toself',
                fillcolor=fill_color,
                line=dict(color='rgba(255,255,255,0)'),
                hoverinfo='skip',
                showlegend=False,
                legendgroup=canon,  # group band+line per variant
            ))

        # Line (legend entry)
        fig.add_trace(go.Scatter(
            x=x, y=y_mean,
            mode='lines+markers',
            name=name,
            line=dict(color=color_hex, width=2.5),
            marker=dict(symbol=marker, size=8),
            legendgroup=canon,  # same variant group
        ))

    # Reorder legend/groups: keep only present
    desired = [k for k in desired_legend_order if k in groups_present]
    _reorder_plotly_legend(fig, desired)

    sel_txt = f"top-1 by {selection_metric}; tie-break (score ↓ → earliest ↑ → stability ↓)"
    fig.update_layout(
        legend=dict(groupclick="togglegroup", orientation="h",
                    x=0.5, xanchor="center", y=-0.12, yanchor="top", traceorder="grouped"),
        title=format_title("Performance of Top-1 HP Trial",
                           f"on {format_env_name(env)} - {sel_txt}", backend="plotly"),
        xaxis_title="Timesteps",
        yaxis_title="Mean Return (± Std Dev)",
        margin=dict(l=60, r=20, t=90, b=140),
        height=650
    )

    filename = get_plot_filename("top1_run_comparison", env)
    save_figure(fig, output_dir, filename)


def plot_aggregate_iqm_plotly(
    env: str,
    algo_agg_dict: dict[str, tuple],
    output_dir: Path,
    top_n: int = 5,
):
    """Interactive aggregate IQM with grouped CI bands and ordered legend; variants plotted separately."""
    fig = go.Figure()
    style = DEFAULT_STYLE

    present_labels = list(algo_agg_dict.keys())
    families, fas = present_families_fas_from_labels(present_labels, FAMILY_ORDER, FA_ORDER)

    def canon_from_label(lbl: str) -> str:
        fam, fa, var = split_family_fa(lbl)
        base = f"{fam}-{fa}" if fa else fam
        return f"{base} ({var})" if var else base

    all_canons = [canon_from_label(l) for l in present_labels]

    def ordered_canons():
        out = []
        if fas:
            for fam in families:
                for fa in fas:
                    base = f"{fam}-{fa}"
                    group = [c for c in all_canons if c.startswith(base)]
                    bare = [base] if base in group else []
                    vars_ = sorted([c for c in group if c != base])
                    out.extend(bare + vars_)
        else:
            for fam in families:
                base = fam
                group = [c for c in all_canons if c == base or c.startswith(f"{base} (")]
                bare = [base] if base in group else []
                vars_ = sorted([c for c in group if c != base])
                out.extend(bare + vars_)
        return out

    desired_legend_order = ordered_canons()
    groups_present = set()

    for algo, (timesteps, iqm, ci_low, ci_high) in algo_agg_dict.items():
        fam, fa, _variant = split_family_fa(algo)
        color_hex = style.color_for(fam, fa)
        fill_color = _rgba_fill(color_hex, 0.2)
        marker = style.plotly_marker_for_fa(fa)

        canon = canon_from_label(algo)  # unique per variant
        groups_present.add(canon)

        # CI band
        fig.add_trace(go.Scatter(
            x=np.concatenate([timesteps, timesteps[::-1]]),
            y=np.concatenate([ci_high, ci_low[::-1]]),
            fill='toself',
            fillcolor=fill_color,
            line=dict(color='rgba(255,255,255,0)'),
            hoverinfo='skip',
            showlegend=False,
            legendgroup=canon,
        ))

        # IQM line
        fig.add_trace(go.Scatter(
            x=timesteps, y=iqm,
            mode='lines+markers',
            name=format_algo_name(algo),
            line=dict(color=color_hex, width=2.5),
            marker=dict(symbol=marker, size=8),
            legendgroup=canon,
        ))

    desired = [k for k in desired_legend_order if k in groups_present]
    _reorder_plotly_legend(fig, desired)

    fig.update_layout(
        legend=dict(groupclick="togglegroup", orientation="h",
                    x=0.5, xanchor="center", y=-0.12, yanchor="top", traceorder="grouped"),
        title=format_title(f"Aggregate Performance of Top-{top_n} HP Trials",
                           f"on {format_env_name(env)} (IQM with 95% CI)", backend="plotly"),
        xaxis_title="Timesteps",
        yaxis_title=f"Return (IQM)",
        margin=dict(l=60, r=20, t=90, b=140),
        height=650
    )

    filename = get_plot_filename(f"top{top_n}_aggregate_perf", env, suffix="iqm")
    save_figure(fig, output_dir, filename)


def plot_slice_plotly(
    env: str,
    algo_slice_dict: dict[str, pd.DataFrame],
    output_dir: Path,
    *,
    metric_column: str = "peak_return",
):
    """
    Interactive Optuna-style slice plots for hyperparameter sensitivity.

    Args:
        env: Environment name.
        algo_slice_dict: {algo: tidy dataframe with ['hp_id','param','value',metric_column]}.
        output_dir: Destination directory.
        metric_column: Summary metric used for the Y axis (default: 'peak_return').
    """
    if not algo_slice_dict:
        return

    metric_pretty = metric_column.replace("_", " ").title()

    for algo, df in algo_slice_dict.items():
        if df is None or df.empty:
            continue

        required = {"param", "value", metric_column}
        if not required.issubset(df.columns):
            continue

        df_clean = df.dropna(subset=["value", metric_column]).copy()
        if df_clean.empty:
            continue

        params = [
            param
            for param, group in df_clean.groupby("param")
            if group["value"].nunique(dropna=True) >= 2
        ]
        if not params:
            continue
        params.sort()

        n_params = len(params)
        ncols = n_params
        nrows = 1

        metric_values = df_clean[metric_column].to_numpy(dtype=float)
        metric_values = metric_values[np.isfinite(metric_values)]
        variable_color = metric_values.size > 0 and not np.allclose(
            metric_values.max(), metric_values.min()
        )
        color_min = float(metric_values.min()) if metric_values.size else 0.0
        color_max = float(metric_values.max()) if metric_values.size else 1.0

        if ncols == 1:
            spacing = 0.0
            subplot_titles = [format_hyperparam_name(params[0])]
        else:
            spacing = min(0.02, max(0.01, 0.2 / ncols))
            subplot_titles = [format_hyperparam_name(p) for p in params]

        fig = make_subplots(
            rows=1,
            cols=ncols,
            subplot_titles=subplot_titles,
            horizontal_spacing=spacing,
        )

        showscale = variable_color

        for idx, param in enumerate(params):
            subset = df_clean[df_clean["param"] == param]
            x_raw = subset["value"].to_numpy(dtype=float)
            y_raw = subset[metric_column].to_numpy(dtype=float)
            hp_ids = subset["hp_id"].to_numpy()
            mask = np.isfinite(x_raw) & np.isfinite(y_raw)
            x = x_raw[mask]
            y = y_raw[mask]
            hp_ids = hp_ids[mask]
            if x.size == 0:
                continue

            row = 1
            col = idx + 1
            pretty_param = format_hyperparam_name(param)

            hover_text = [
                f"hp_id={hp}<br>{pretty_param}={val:.4g}<br>{metric_pretty}={score:.4g}"
                for hp, val, score in zip(hp_ids, x, y)
            ]

            marker_kwargs: dict = dict(
                size=9,
                line=dict(width=0.5, color="rgba(0,0,0,0.45)"),
                opacity=0.9,
            )

            if variable_color:
                marker_kwargs.update(
                    color=y,
                    colorscale="Plotly3",
                    cmin=color_min,
                    cmax=color_max,
                    showscale=showscale,
                )
                if showscale:
                    marker_kwargs["colorbar"] = dict(
                        title=metric_pretty,
                        len=0.8,
                        y=0.5,
                        thickness=16,
                    )
                    showscale = False
            else:
                marker_kwargs["color"] = "rgba(31,119,180,0.85)"

            fig.add_trace(
                go.Scatter(
                    x=x,
                    y=y,
                    mode="markers",
                    hovertemplate="%{text}<extra></extra>",
                    text=hover_text,
                    showlegend=False,
                    marker=marker_kwargs,
                ),
                row=row,
                col=col,
            )

            values = x[np.isfinite(x)]
            if values.size > 0 and np.all(values > 0):
                x_min = float(values.min())
                x_max = float(values.max())
                if x_min > 0 and (x_max / max(x_min, 1e-12)) >= 100.0:
                    fig.update_xaxes(type="log", row=row, col=col)

            fig.update_xaxes(title_text=pretty_param, row=row, col=col)
            if col == 1:
                fig.update_yaxes(title_text=metric_pretty, row=row, col=col)
            else:
                fig.update_yaxes(title_text="", row=row, col=col)

        height = 480
        width = max(900, int(480 * ncols))
        fig.update_layout(
            title=format_title(
                f"Hyperparameter Slice Plot - {format_algo_name(algo)}",
                f"on {format_env_name(env)} - {metric_pretty}",
                backend="plotly",
            ),
            margin=dict(l=70, r=80, t=90, b=70),
            height=height,
            width=width,
            hovermode="closest",
        )
        fig.update_xaxes(showgrid=True, zeroline=False)
        fig.update_yaxes(showgrid=True, zeroline=False)

        filename = get_plot_filename(
            "slice_plot",
            env,
            algo=format_algo_name(algo),
            suffix=metric_column,
        )
        save_figure(fig, output_dir, filename)
