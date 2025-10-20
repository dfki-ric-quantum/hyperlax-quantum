#!/usr/bin/env python3
import argparse
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from hyperlax.analysis.runtime_metadata import (
    aggregate_timing_records,
    collect_timing_records,
    export_metadata_csv,
)

def _tune_layout(fig, w_pad=0.6, h_pad=0.7, wspace=0.35, hspace=0.45, suptitle_y=None):
    """
    Minimal, robust spacing strategy:
      - Use constrained layout and nudge the global pads slightly.
    """
    # Leave suptitle positioning to constrained layout
    if suptitle_y is not None and fig._suptitle is not None:
        fig._suptitle.set_y(suptitle_y)
    fig.set_constrained_layout_pads(w_pad=w_pad, h_pad=h_pad, wspace=wspace, hspace=hspace)

def _add_experiment_caption(fig: plt.Figure, metadata: dict, fontsize:int=9) -> None:
    parts = []
    if metadata.get("model") not in (None, "", "N/A"):
        parts.append(f"Model: {metadata['model']}")
    if metadata.get("env") not in (None, "", "N/A"):
        parts.append(f"Environment: {metadata['env']}")
    gpu = metadata.get("gpu")
    setup = metadata.get("setup")
    if gpu not in (None, "", "N/A"):
        if setup not in (None, "", "N/A"):
            parts.append(f"GPU: {gpu} ({setup})")
        else:
            parts.append(f"GPU: {gpu}")
    elif setup not in (None, "", "N/A"):
        parts.append(f"Setup: {setup}")

    caption_text = " | ".join(parts)
    if caption_text:
        fig.text(
            0.5, 0.01, caption_text,
            ha="center", va="bottom", fontsize=fontsize,
            bbox=dict(facecolor="white", alpha=0.5, edgecolor="none"),
        )


def _format_seconds_hms(seconds: float) -> str:
    if not np.isfinite(seconds): return "N/A"
    seconds = int(round(seconds))
    h, rem = divmod(seconds, 3600)
    m, s = divmod(rem, 60)
    return f"{h:02}:{m:02}:{s:02}" if h else f"{m:02}:{s:02}"

def _mode_color_marker(mode: str) -> Tuple[str, str]:
    if mode == "Sequential": return ("red", "x")
    if mode == "Naive Vectorized": return ("orange", "s")
    if mode == "Grouped Vectorized": return ("green", "o")
    if mode == "Vectorized": return ("C0", "o")
    return ("C7", "o")

def _ensure_plots_dir(out_dir: Path) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)

def _variant_styles(variants: List[str]) -> Dict[str, Dict[str, object]]:
    """
    Returns a dict: variant -> {"alpha": float, "hatch": str, "label": str}
    Empty variant "" is treated as 'base'.
    """
    # bounded alphas to keep visible
    max_drop = 0.5
    n = max(1, len(variants))
    steps = np.linspace(0.0, max_drop, n)
    hatches = ["", "///", "\\\\\\", "xxx", "...", "+++"]
    styles = {}
    for i, v in enumerate(variants):
        styles[v] = {
            "alpha": float(1.0 - steps[i]),
            "hatch": hatches[i % len(hatches)],
            "label": (v if v else "base"),
        }
    return styles

def plot_summary_by_batch_per_model(df: pd.DataFrame, out_root: Path, caption_defaults: dict) -> None:
    if df.empty:
        print("No data for batch-preserving summary.")
        return

    df = df.copy()
    df.loc[df["run_mode"] == "Sequential", "batch_size"] = 1
    df["batch_size"] = pd.to_numeric(df["batch_size"], errors="coerce")

    # Baseline per (model, config_id) so variants never mix
    base = (
        df[df["run_mode"] == "Sequential"][["model", "config_id", "total_wall_time"]]
        .rename(columns={"total_wall_time": "seq_time"})
    )
    merged = df.merge(base, on=["model", "config_id"], how="left")
    merged["speedup"] = merged["seq_time"] / merged["total_wall_time"]

    modes_order = ["Sequential", "Naive Vectorized", "Grouped Vectorized", "Vectorized"]

    for exp_type in sorted(merged["exp_type"].dropna().unique()):
        fam = merged[merged["exp_type"] == exp_type]
        for model in sorted(fam["model"].dropna().unique()):
            df_m = fam[fam["model"] == model].copy()
            if df_m.empty or df_m["seq_time"].isna().all():
                print(f"[summary] Missing sequential baseline for {model}/{exp_type}; skipping.")
                continue

            # Variants present for this (model, exp_type)
            variants = sorted(df_m["exp_variant"].fillna("").unique().tolist())
            vstyles = _variant_styles(variants)

            # Batches to show = union across everything for this model/family
            batches = sorted([int(b) for b in df_m["batch_size"].dropna().unique()])
            if not batches:
                continue

            fig = plt.figure(figsize=(max(8, 1.2 * len(batches)), 6), constrained_layout=True)
            ax = fig.add_subplot(111)

            x = np.arange(len(batches), dtype=float)
            present_modes = [m for m in modes_order if m in df_m["run_mode"].unique()]
            n_modes = max(1, len([m for m in present_modes if m != "Sequential"]))
            width = 0.8 / max(n_modes, 1)

            slot_idx = 0
            for mode in present_modes:
                if mode == "Sequential":
                    continue
                col, _marker = _mode_color_marker(mode)
                offsets = x - 0.4 + width/2 + slot_idx * width
                slot_idx += 1

                # For each batch, we may have multiple variants. Center and subdivide the bar.
                for i, b in enumerate(batches):
                    sub = df_m[(df_m["run_mode"] == mode) & (df_m["batch_size"] == b)].copy()
                    if sub.empty:
                        continue
                    sub["exp_variant"] = sub["exp_variant"].fillna("")
                    present_vs = [v for v in variants if v in sub["exp_variant"].values]
                    k = max(1, len(present_vs))
                    vwidth = width / k
                    start = offsets[i] - width/2
                    for j, v in enumerate(present_vs):
                        row = sub[sub["exp_variant"] == v]
                        sp = float(row["speedup"].iloc[0])
                        rt = float(row["total_wall_time"].iloc[0])
                        xpos = start + j * vwidth + vwidth/2
                        style = vstyles[v]
                        ax.bar([xpos], [sp], width=vwidth, color=col, alpha=style["alpha"], hatch=style["hatch"])
                        ax.annotate(f"{sp:.1f}x ({_format_seconds_hms(rt)}, B={b})",
                                    xy=(xpos, sp), xytext=(0, 4), textcoords="offset points",
                                    ha="center", va="bottom", fontsize=8, rotation=90)

            ax.axhline(1.0, ls="--", color="red", linewidth=1.0, label="Sequential (1×)")
            if 1 in batches:
                i = batches.index(1)
                ax.scatter([x[i]], [1.0], s=35, color="red", zorder=3)

            # Cosmetics
            ax.set_xticks(x)
            ax.set_xticklabels([str(b) for b in batches])
            ax.set_xlabel("Batch Size")
            ax.set_ylabel("Speedup vs. Sequential (↑ better)")
            title_map = {
                "homogeneous_best": "Homogeneous (Best Case)",
                "hetero_algo": "Heterogeneous (Algo HPs)",
                "hetero_network": "Heterogeneous (Network HPs)",
                "hetero_algo_network": "Heterogeneous (Algo + Network HPs)",
                "unknown": "Unknown Experiment",
            }
            ax.set_title(f"Summary — All Batch Sizes by Mode\n{title_map.get(exp_type, exp_type)} — {model}")
            ax.grid(True, axis="y", linestyle="--", linewidth=0.5)

            # variant legend (outside, top-right)
            handles, labels = [], []
            for v in variants:
                style = vstyles[v]
                patch = plt.Rectangle((0,0),1,1, facecolor="gray", alpha=style["alpha"], hatch=style["hatch"])
                handles.append(patch)
                labels.append(style["label"])
            if handles:
                fig.legend(handles, labels, title="Variant", ncol=min(4, len(handles)),
                           loc="upper right", bbox_to_anchor=(0.98, 0.995),
                           fontsize=9, title_fontsize=10, frameon=False)

            env_uniques = df_m["env"].dropna().unique()
            env = env_uniques[0] if len(env_uniques) else caption_defaults.get("env", "N/A")
            cap = dict(caption_defaults); cap.update({"model": model, "env": env})
            _add_experiment_caption(fig, cap)

            _tune_layout(fig)
            out_dir = out_root / "summary_by_batch" / model
            _ensure_plots_dir(out_dir)
            p = out_dir / f"{model}_{exp_type}_summary_by_batch.png"
            fig.savefig(p, dpi=300, bbox_inches="tight"); plt.close(fig)
            print(f"Saved: {p}")

def plot_summary_by_batch_all_models(df: pd.DataFrame, out_root: Path, caption_defaults: dict, cols: int = 3) -> None:
    if df.empty:
        return

    df = df.copy()
    df.loc[df["run_mode"] == "Sequential", "batch_size"] = 1
    df["batch_size"] = pd.to_numeric(df["batch_size"], errors="coerce")

    base = df[df["run_mode"] == "Sequential"][["model", "config_id", "total_wall_time"]].rename(
        columns={"total_wall_time": "seq_time"}
    )
    merged = df.merge(base, on=["model", "config_id"], how="left")
    merged["speedup"] = merged["seq_time"] / merged["total_wall_time"]

    modes_order = ["Naive Vectorized", "Grouped Vectorized", "Vectorized"]

    for exp_type in sorted(merged["exp_type"].dropna().unique()):
        fam = merged[merged["exp_type"] == exp_type].copy()
        models = sorted(fam["model"].unique())
        if not models:
            continue

        # discover variants present overall to build consistent legend styles
        variants = sorted(fam["exp_variant"].fillna("").unique().tolist())
        vstyles = _variant_styles(variants)

        rows = int(np.ceil(len(models) / cols))
        fig, axes = plt.subplots(
            rows, cols, figsize=(max(10, 4 * cols), max(6, 3.5 * rows)),
            constrained_layout=True
        )

        axes = np.atleast_2d(axes)
        for idx, model in enumerate(models, start=1):
            r = (idx - 1) // cols; c = (idx - 1) % cols
            ax = axes[r, c]
            df_m = fam[fam["model"] == model].copy()
            if df_m.empty or df_m["seq_time"].isna().all():
                ax.set_visible(False); continue

            batches = sorted([int(b) for b in df_m["batch_size"].dropna().unique()])
            x = np.arange(len(batches), dtype=float)

            present_modes = [m for m in modes_order if m in df_m["run_mode"].unique()]
            n_modes = max(1, len(present_modes)); width = 0.8 / n_modes

            for s, mode in enumerate(present_modes):
                col, _ = _mode_color_marker(mode)
                offsets = x - 0.4 + width/2 + s * width

                # For each batch, subdivide the bar by variant
                for i, b in enumerate(batches):
                    sub = df_m[(df_m["run_mode"] == mode) & (df_m["batch_size"] == b)].copy()
                    if sub.empty:
                        continue
                    sub["exp_variant"] = sub["exp_variant"].fillna("")
                    present_vs = [v for v in variants if v in sub["exp_variant"].values]
                    k = max(1, len(present_vs))
                    vwidth = width / k
                    start = offsets[i] - width/2
                    for j, v in enumerate(present_vs):
                        row = sub[sub["exp_variant"] == v]
                        sp = float(row["speedup"].iloc[0])
                        xpos = start + j * vwidth + vwidth/2
                        style = vstyles[v]
                        ax.bar([xpos], [sp], width=vwidth, color=col, alpha=style["alpha"], hatch=style["hatch"])

            # Baseline
            ax.axhline(1.0, ls="--", color="red", linewidth=1.0)
            if 1 in batches:
                i = batches.index(1)
                ax.scatter([x[i]], [1.0], s=25, color="red", zorder=3)

            ax.set_title(model, fontsize=10)
            ax.set_xticks(x)
            ax.set_xticklabels([str(b) for b in batches])#, rotation=0)
            ymax = np.nanmax(df_m["speedup"].to_numpy(dtype=float))
            ax.set_ylim(0, max(2.0, 1.15 * (ymax if np.isfinite(ymax) else 2.0)))
            ax.grid(True, axis="y", linestyle="--", linewidth=0.5)

        # clear any unused axes
        total = rows * cols
        for idx in range(len(models)+1, total+1):
            r = (idx - 1) // cols; c = (idx - 1) % cols
            axes[r, c].set_visible(False)

        title_map = {
            "homogeneous_best": "Summary — All Batch Sizes by Mode (Homogeneous)",
            "hetero_algo": "Summary — All Batch Sizes by Mode (Heterogeneous: Algo HPs)",
            "hetero_network": "Summary — All Batch Sizes by Mode (Heterogeneous: Network HPs)",
            "hetero_algo_network": "Summary — All Batch Sizes by Mode (Heterogeneous: Algo + Network HPs)",
        }
        fig.suptitle(title_map.get(exp_type, f"Summary — All Batch Sizes by Mode ({exp_type})"), fontsize=14)

        # variant legend outside, bottom-center
        handles, labels = [], []
        for v in variants:
            style = vstyles[v]
            patch = plt.Rectangle((0,0),1,1, facecolor="gray", alpha=style["alpha"], hatch=style["hatch"])
            handles.append(patch); labels.append(style["label"])
        if handles:
            fig.legend(handles, labels, title="Variant", ncol=min(4, len(handles)),
                       loc="lower center", bbox_to_anchor=(0.5, 0.02),
                       fontsize=9, title_fontsize=10, frameon=False)

        _add_experiment_caption(fig, dict(caption_defaults))
        _tune_layout(fig, h_pad=0.9, hspace=0.5)

        out_dir = out_root / "summary_by_batch"
        _ensure_plots_dir(out_dir)
        p = out_dir / f"summary_by_batch_grid_{exp_type}.png"
        fig.savefig(p, dpi=300, bbox_inches="tight"); plt.close(fig)
        print(f"Saved: {p}")

def compute_global_speedup_max(df: pd.DataFrame, margin: float = 1.15, floor: float = 2.0) -> float:
    d = df.copy()
    d.loc[d["run_mode"] == "Sequential", "batch_size"] = 1
    d["batch_size"] = pd.to_numeric(d["batch_size"], errors="coerce")

    base = (
        d[d["run_mode"] == "Sequential"][["model", "config_id", "total_wall_time"]]
        .rename(columns={"total_wall_time": "seq_time"})
    )
    m = d.merge(base, on=["model", "config_id"], how="left")
    m["speedup"] = m["seq_time"] / m["total_wall_time"]

    sp = m["speedup"].astype(float)
    sp = sp[np.isfinite(sp)]
    if sp.empty:
        return floor

    ymax = float(sp.max()) * margin
    return max(floor, ymax)

def plot_final_summary_grid_speedup(
    df: pd.DataFrame,
    out_root: Path,
    caption_defaults: dict,
    cols: Optional[List[str]] = None,
    y_max: Optional[float] = None,  # shared speedup ceiling (left axis)
) -> None:
    if df.empty:
        print("No data for final speedup grid.")
        return

    def _fmt(seconds: float, show_hours: bool) -> str:
        if not np.isfinite(seconds) or seconds < 0: return ""
        seconds = int(round(seconds)); h, rem = divmod(seconds, 3600); m, s = divmod(rem, 60)
        return f"{h:02}:{m:02}:{s:02}" if show_hours else f"{m:02}:{s:02}"

    data = df.copy()
    data.loc[data["run_mode"] == "Sequential", "batch_size"] = 1
    data["batch_size"] = pd.to_numeric(data["batch_size"], errors="coerce")

    base = data[data["run_mode"] == "Sequential"][["model","config_id","total_wall_time"]].rename(
        columns={"total_wall_time": "seq_time"}
    )
    merged = data.merge(base, on=["model","config_id"], how="left")
    merged["speedup"] = merged["seq_time"] / merged["total_wall_time"]

    all_rt = np.concatenate([merged["seq_time"].to_numpy(float), merged["total_wall_time"].to_numpy(float)])
    has_hour_runtime = np.isfinite(all_rt).any() and np.nanmax(all_rt) >= 3600.0

    # Layout
    title_map = {
        "homogeneous_best": "Homogeneous",
        "hetero_algo": "Hetero: Algo HPs",
        "hetero_network": "Hetero: Network HPs",
        "hetero_algo_network": "Hetero: Algo+Network",
        "unknown": "Unknown",
    }
    if cols is None:
        pref = ["homogeneous_best","hetero_algo","hetero_network","hetero_algo_network","unknown"]
        present = [e for e in pref if e in merged["exp_type"].dropna().unique().tolist()]
        cols = present if present else sorted(merged["exp_type"].dropna().unique().tolist())

    models = sorted(merged["model"].dropna().unique().tolist())
    if not models or not cols:
        print("Nothing to plot (no models or exp_types)."); return

    # discover variants present overall to build consistent legend styles
    variants = sorted(merged["exp_variant"].fillna("").unique().tolist())
    vstyles = _variant_styles(variants)
    modes_order = ["Naive Vectorized","Grouped Vectorized","Vectorized"]

    # figure with constrained layout ON
    n_rows, n_cols = len(models), len(cols)
    fig, axes = plt.subplots(
        n_rows, n_cols,
        figsize=(max(10, 4.3*n_cols), max(6, 3.8*n_rows)),
        squeeze=False, gridspec_kw={"wspace": 0.35, "hspace": 0.35},
        constrained_layout=True
    )
    # We'll set column headers on the top row axes (inside axes area)
    col_headers = [title_map.get(e, e) for e in cols]
    for c, header in enumerate(col_headers):
        top_ax = axes[0][c]
        top_ax.set_title(header, fontsize=11, fontweight="bold", pad=10, loc="center")

    present_modes_global = set(merged["run_mode"].dropna().unique().tolist())
    has_grouped = "Grouped Vectorized" in present_modes_global

    import matplotlib.patches as mpatches
    from matplotlib.lines import Line2D

    if has_grouped:
        no_group_color, _ = _mode_color_marker("Naive Vectorized")
        group_color, _    = _mode_color_marker("Grouped Vectorized")
        # (legend rendered later, outside)
    else:
        no_group_color = "C0"; group_color = None

    def color_for(mode: str, exp_type: str) -> str:
        if has_grouped:
            if mode == "Grouped Vectorized":
                return group_color
            if (exp_type == "homogeneous_best" and mode == "Vectorized") or (mode == "Naive Vectorized"):
                return no_group_color
            col, _mk = _mode_color_marker(mode)
            return col
        return no_group_color

    for r, model in enumerate(models):
        for c, exp_type in enumerate(cols):
            ax = axes[r][c]
            sub = merged[(merged["model"]==model)&(merged["exp_type"]==exp_type)].copy()
            if sub.empty or sub["seq_time"].isna().all():
                ax.set_visible(False); continue

            batches = sorted([int(b) for b in sub["batch_size"].dropna().unique()])
            x = np.arange(len(batches), dtype=float)
            present_modes = [m for m in modes_order if m in sub["run_mode"].unique()]

            local_max = 1.0; base_span = 0.8
            for i, b in enumerate(batches):
                for mode in present_modes:
                    rows = sub[(sub["run_mode"]==mode) & (sub["batch_size"]==b)]
                    if rows.empty:
                        continue
                    rows = rows.copy()
                    rows["exp_variant"] = rows["exp_variant"].fillna("")
                    present_vs = [v for v in variants if v in rows["exp_variant"].values]
                    k = max(1, len(present_vs))
                    width = base_span / (len(present_modes) * k)
                    # offset start so modes occupy contiguous chunks around the tick
                    start = x[i] - base_span/2 + present_modes.index(mode) * (base_span/len(present_modes))
                    for j, v in enumerate(present_vs):
                        row = rows[rows["exp_variant"] == v]
                        if row.empty:
                            continue
                        sp = float(row["speedup"].iloc[0])
                        xpos = start + (j + 0.5) * width
                        col = color_for(mode, exp_type)
                        style = vstyles[v]
                        ax.bar([xpos], [sp], width=width, color=col, alpha=style["alpha"], hatch=style["hatch"])
                        local_max = max(local_max, sp)

            # axes
            ax.axhline(1.0, ls="--", color="red", linewidth=1.0)
            ax.set_xticks(x); ax.set_xticklabels([str(b) for b in batches], fontsize=8)
            ax.set_xlabel("HP Batch Sizes", fontsize=9)
            ax.grid(True, axis="y", linestyle="--", linewidth=0.5)
            ax.set_ylim(0, y_max if y_max is not None else max(2.0, local_max*1.18))
            if c == 0: ax.set_ylabel(f"{model}\nSpeedup (×)", fontsize=10)

            # right axis runtime mirror
            seq_time = float(sub["seq_time"].iloc[0]) if not sub["seq_time"].isna().all() else None
            if seq_time and np.isfinite(seq_time) and seq_time>0:
                ax_rt = ax.twinx()
                ax_rt.set_ylim(ax.get_ylim())
                yt = list(ax.get_yticks())
                if 1.0 not in yt:
                    yt.append(1.0); yt = sorted(yt)
                rt_labels = []
                for t in yt:
                    if t<=0 or not np.isfinite(t): rt_labels.append("")
                    else:
                        rt_labels.append(_fmt(seq_time/float(t), has_hour_runtime))
                ax_rt.set_yticks(yt); ax_rt.set_yticklabels(rt_labels, fontsize=8)

    fig.suptitle(
        "Runtime Scaling across Hyperparameter Batch Sizes\n"
        "Bars: Speedup (left) • Right axis: Runtime\n"
        "Rows: Models • Columns: Experiment Families",
        fontsize=12
    )

    # variant legend (outside, top-right)
    handles_v, labels_v = [], []
    for v in variants:
        style = vstyles[v]
        patch = plt.Rectangle((0,0),1,1, facecolor="gray", alpha=style["alpha"], hatch=style["hatch"])
        handles_v.append(patch); labels_v.append(style["label"])
    if handles_v:
        fig.legend(handles_v, labels_v, #title="Variant",
                   ncol=1,
                   loc="upper right", bbox_to_anchor=(0.98, 0.995),
                   fontsize=9, title_fontsize=10, frameon=False)

    _add_experiment_caption(fig, dict(caption_defaults))
    _tune_layout(fig, h_pad=0.5, hspace=0.2)

    out_dir = out_root / "summary"; _ensure_plots_dir(out_dir)
    p = out_dir / "final_summary_grid_speedup.png"
    fig.savefig(p, dpi=300, bbox_inches="tight"); plt.close(fig)
    print(f"Saved: {p}")

def compute_global_runtime_max(df: pd.DataFrame, margin: float = 1.12, floor: float = 1.0) -> float:
    """
    Returns a y-axis max (in seconds) so ALL runtime plots share the same range.
    margin: multiplicative headroom on the max total_wall_time.
    floor:  minimum y max to avoid tiny axes.
    """
    if df.empty or "total_wall_time" not in df.columns:
        return floor
    vals = pd.to_numeric(df["total_wall_time"], errors="coerce")
    vals = vals[np.isfinite(vals)]
    if vals.empty:
        return floor
    ymax = float(vals.max()) * margin
    return max(floor, ymax)

def plot_final_summary_grid_runtime_stacked(
    df: pd.DataFrame,
    out_root: Path,
    caption_defaults: dict,
    cols: Optional[List[str]] = None,
    y_max: Optional[float] = None,   # shared runtime ceiling
) -> None:
    if df.empty:
        print("No data for final runtime grid.")
        return

    def _fmt(seconds: float, show_hours: bool) -> str:
        if not np.isfinite(seconds) or seconds < 0: return ""
        seconds = int(round(seconds)); h, rem = divmod(seconds, 3600); m, s = divmod(rem, 60)
        return f"{h:02}:{m:02}:{s:02}" if show_hours else f"{m:02}:{s:02}"

    data = df.copy()
    data.loc[data["run_mode"] == "Sequential", "batch_size"] = 1
    data["batch_size"] = pd.to_numeric(data["batch_size"], errors="coerce")
    if "jit_time" not in data.columns: data["jit_time"] = np.nan
    if "execution_time" not in data.columns: data["execution_time"] = np.nan
    data["jit_time"] = data["jit_time"].fillna(0.0)
    data["execution_time"] = data["execution_time"].where(
        data["execution_time"].notna(), (data["total_wall_time"] - data["jit_time"]).clip(lower=0.0)
    )

    base = data[data["run_mode"] == "Sequential"][["model","config_id","total_wall_time"]].rename(
        columns={"total_wall_time":"seq_time"}
    )
    merged = data.merge(base, on=["model","config_id"], how="left")

    all_rt = merged["total_wall_time"].to_numpy(float)
    has_hour_runtime = np.isfinite(all_rt).any() and np.nanmax(all_rt) >= 3600.0
    title_map = {
        "homogeneous_best": "Homogeneous",
        "hetero_algo": "Hetero: Algo HPs",
        "hetero_network": "Hetero: Network HPs",
        "hetero_algo_network": "Hetero: Algo+Network",
        "unknown": "Unknown",
    }
    if cols is None:
        pref = ["homogeneous_best","hetero_algo","hetero_network","hetero_algo_network","unknown"]
        present = [e for e in pref if e in merged["exp_type"].dropna().unique().tolist()]
        cols = present if present else sorted(merged["exp_type"].dropna().unique().tolist())

    models = sorted(merged["model"].dropna().unique().tolist())
    if not models or not cols:
        print("Nothing to plot (no models or exp_types)."); return

    # variants for legend
    variants = sorted(merged["exp_variant"].fillna("").unique().tolist())
    vstyles = _variant_styles(variants)
    modes_order = ["Sequential","Naive Vectorized","Grouped Vectorized","Vectorized"]

    n_rows, n_cols = len(models), len(cols)
    fig, axes = plt.subplots(
        n_rows, n_cols,
        figsize=(max(10, 4.6*n_cols), max(6, 4.0*n_rows)),
        squeeze=False, gridspec_kw={"wspace": 0.42, "hspace": 0.42},
        constrained_layout=True
    )

    col_headers = [title_map.get(e, e) for e in cols]
    for c, header in enumerate(col_headers):
        top_ax = axes[0][c]
        top_ax.set_title(header, fontsize=11, fontweight="bold", pad=10, loc="center")

    present_modes_global = set(merged["run_mode"].dropna().unique().tolist())
    has_grouped = "Grouped Vectorized" in present_modes_global

    import matplotlib.patches as mpatches

    def color_for(mode: str, exp_type: str) -> str:
        if has_grouped:
            if mode == "Sequential": return "0.6"
            if mode == "Grouped Vectorized": return _mode_color_marker("Grouped Vectorized")[0]
            if (exp_type == "homogeneous_best" and mode == "Vectorized") or (mode == "Naive Vectorized"):
                return _mode_color_marker("Naive Vectorized")[0]
            return _mode_color_marker(mode)[0]
        return "0.6" if mode == "Sequential" else "C0"

    for r, model in enumerate(models):
        for c, exp_type in enumerate(cols):
            ax = axes[r][c]
            sub = merged[(merged["model"]==model)&(merged["exp_type"]==exp_type)].copy()
            if sub.empty or sub["seq_time"].isna().all():
                ax.set_visible(False); continue

            batches = sorted([int(b) for b in sub["batch_size"].dropna().unique()])
            x = np.arange(len(batches), dtype=float)
            present_modes = [m for m in modes_order if m in sub["run_mode"].unique()]

            ymax_rt = 0.0; base_span = 0.8
            for i, b in enumerate(batches):
                for mode in present_modes:
                    rows = sub[(sub["run_mode"]==mode) & (sub["batch_size"]==b)]
                    if rows.empty: continue
                    rows = rows.copy()
                    rows["exp_variant"] = rows["exp_variant"].fillna("")
                    present_vs = [v for v in variants if v in rows["exp_variant"].values]
                    k = max(1, len(present_vs))
                    width = base_span / (len(present_modes) * k)
                    start = x[i] - base_span/2 + present_modes.index(mode) * (base_span/len(present_modes))
                    for j, v in enumerate(present_vs):
                        row = rows[rows["exp_variant"] == v]
                        if row.empty: continue
                        tot = float(row["total_wall_time"].iloc[0])
                        ex  = float(row["execution_time"].iloc[0])
                        jt  = float(row["jit_time"].iloc[0])
                        # Fix mild inconsistencies
                        if not np.isfinite(ex): ex = max(0.0, tot - jt)
                        if not np.isfinite(jt): jt = max(0.0, tot - ex)
                        xpos = start + (j + 0.5) * width
                        col = color_for(mode, exp_type)
                        style = vstyles[v]
                        JIT_HATCH = ".."  # fixed overlay pattern for JIT (distinct from most variant hatches)

                        # execution base: mode color + VARIANT HATCH (primary cue), no alpha dimming
                        ax.bar([xpos], [ex],
                               width=width,
                               color=col,
                               hatch=style["hatch"],   # << variant signal
                               alpha=style["alpha"]
                               )

                        # JIT overlay: transparent fill, colored edge, FIXED hatch (not tied to variant)
                        ax.bar([xpos], [jt],
                               width=width,
                               bottom=[ex],
                               facecolor="none",
                               edgecolor=col,
                               hatch=JIT_HATCH,        # << consistent across variants
                               linewidth=0.0)

                        ymax_rt = max(ymax_rt, ex + jt)

            # NO dashed sequential line here — Sequential has its own stacked bar at B=1.

            ax.set_ylim(0, y_max if (y_max is not None) else max(1e-6, ymax_rt * 1.12))
            yt = ax.get_yticks()
            ax.set_yticklabels([_fmt(t, has_hour_runtime) if (t>0 and np.isfinite(t)) else "" for t in yt], fontsize=8)
            if c == 0: ax.set_ylabel(f"{model}\n{'Runtime (HH:MM:SS)' if has_hour_runtime else 'Runtime (MM:SS)'}", fontsize=10)

            ax.set_xticks(x); ax.set_xticklabels([str(b) for b in batches], fontsize=8)
            ax.set_xlabel("HP Batch Sizes", fontsize=9)
            ax.grid(True, axis="y", linestyle="--", linewidth=0.5)

    fig.suptitle(
        "Runtime (absolute) across Hyperparameter Batch Sizes\n"
        "Bars: Execution (solid) + JIT (striped) • Includes Sequential bar at B=1\n"
        "Rows: Models • Columns: Experiment Families",
        fontsize=12
    )

    # exec/jit pattern legend OUTSIDE, top-right (stacked)
    exec_patch = mpatches.Patch(facecolor="0.7", edgecolor="0.7", label="Execution (solid)")
    jit_patch  = mpatches.Patch(facecolor="none", edgecolor="0.3", hatch="///", label="JIT (striped)")
    fig.legend([exec_patch, jit_patch], [exec_patch.get_label(), jit_patch.get_label()],
               loc="upper right", bbox_to_anchor=(0.98, 0.995), fontsize=9, frameon=False)

    handles_v, labels_v = [], []
    for v in variants:
        style = vstyles[v]  # {"alpha": ..., "hatch": ..., "label": ...}
        # proxy rectangle using neutral facecolor; alpha + hatch encode the variant
        proxy = mpatches.Rectangle((0, 0), 1, 1, facecolor="0.6", alpha=style["alpha"], hatch=style["hatch"])
        handles_v.append(proxy); labels_v.append(style["label"])

    if handles_v:
        fig.legend(handles_v, labels_v, #title="Variant",
                   ncol=1,
                   #ncol=min(4, len(handles_v)),
                   loc="upper right", bbox_to_anchor=(0.98, 0.945),
                   fontsize=9, title_fontsize=10, frameon=False)

    _add_experiment_caption(fig, dict(caption_defaults))
    _tune_layout(fig, h_pad=0.5, hspace=0.2)

    out_dir = out_root / "summary"; _ensure_plots_dir(out_dir)
    p = out_dir / "final_summary_grid_runtime.png"
    fig.savefig(p, dpi=300, bbox_inches="tight"); plt.close(fig)
    print(f"Saved: {p}")

def run_plotting(results_dir: Path, output_dir: Path, gpu_name: str, setup: str, default_env: str) -> None:
    records_df = collect_timing_records(results_dir)
    if records_df.empty:
        print("No results to plot. Exiting.")
        return

    df = aggregate_timing_records(records_df)
    if df.empty:
        print("No aggregated results to plot. Exiting.")
        return

    output_dir.mkdir(parents=True, exist_ok=True)
    summary_csv = output_dir / "runtime_scaling_summary.csv"
    df.to_csv(summary_csv, index=False)
    print(f"Saved summary: {summary_csv}")

    metadata_csv = export_metadata_csv(records_df, summary_csv)
    if metadata_csv:
        print(f"Saved experiment metadata: {metadata_csv}")

    caption_defaults = {
        "gpu": gpu_name,
        "setup": setup,
        "env": default_env,
        "model": "N/A",
    }

    # # Optional per-model summaries
    # plot_summary_by_batch_per_model(df, output_dir, caption_defaults)
    # plot_summary_by_batch_all_models(df, output_dir, caption_defaults, cols=3)

    # One-figure, all-in summaries with shared y ceilings
    global_ymax = compute_global_speedup_max(df, margin=1.15, floor=2.0)
    plot_final_summary_grid_speedup(df, output_dir, caption_defaults, y_max=global_ymax)

    global_rt_ymax = compute_global_runtime_max(df, margin=1.12, floor=1.0)
    plot_final_summary_grid_runtime_stacked(df, output_dir, caption_defaults, y_max=global_rt_ymax)

def main():
    ap = argparse.ArgumentParser(description="Parse and plot runtime scaling benchmark results (variant-aware)")
    ap.add_argument("--results-dir", type=str, required=True, help="Root directory with runtime_scaling.sh outputs")
    ap.add_argument("--output-dir", type=str, default=None, help="Directory to save plots and summary CSV")
    ap.add_argument("--gpu", type=str, default="N/A", help="GPU name for captions (e.g., NVIDIA A100)")
    ap.add_argument("--setup", type=str, default="Single-GPU", help="Setup description for captions")
    ap.add_argument("--default-env", type=str, default="gymnax.pendulum", help="Default env label if not inferred from paths")
    args = ap.parse_args()

    results_dir = Path(args.results_dir)
    output_dir = Path(args.output_dir) if args.output_dir else results_dir / "plots_runtime_scaling"

    run_plotting(results_dir, output_dir, args.gpu, args.setup, args.default_env)

if __name__ == "__main__":
    main()
