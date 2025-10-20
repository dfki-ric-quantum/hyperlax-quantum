"""Statistical aggregation functions (IQM, bootstrap CI, etc.)."""

from typing import Iterable, Optional, Tuple

import numpy as np
import pandas as pd
from scipy.stats import trim_mean

def compute_iqm(data: np.ndarray) -> float:
    """
    Interquartile Mean: mean of middle 50% of data.
    More robust than mean, less extreme than median.
    """
    data_clean = data[~np.isnan(data)]
    if len(data_clean) == 0:
        return np.nan
    return float(trim_mean(data_clean, proportiontocut=0.25))


def bootstrap_ci(
    data: np.ndarray,
    statistic_fn=compute_iqm,
    n_bootstrap: int = 2000,
    alpha: float = 0.05,
) -> tuple[float, float]:
    """
    Compute bootstrap confidence interval for a statistic.

    Args:
        data: 1D array of values
        statistic_fn: Function to compute statistic (default: IQM)
        n_bootstrap: Number of bootstrap samples
        alpha: Confidence level (0.05 = 95% CI)

    Returns:
        (lower, upper) CI bounds
    """
    data_clean = data[~np.isnan(data)]
    if len(data_clean) < 2:
        return np.nan, np.nan

    rng = np.random.default_rng(42)
    bootstrap_stats = []

    for _ in range(n_bootstrap):
        sample = rng.choice(data_clean, size=len(data_clean), replace=True)
        bootstrap_stats.append(statistic_fn(sample))

    lower = np.percentile(bootstrap_stats, alpha / 2 * 100)
    upper = np.percentile(bootstrap_stats, (1 - alpha / 2) * 100)

    return float(lower), float(upper)


def compute_hp_ranking_metrics(
    stats_df: pd.DataFrame,
    metric: str = "peak",
    tail_frac: float = 0.10,
    tol: float = 1e-6,
) -> pd.DataFrame:
    """
    Compute per-hyperparameter ranking statistics (score, earliest reach time, stability).
    """
    required = {"hp_id", "timestep", "mean"}
    if not required.issubset(stats_df.columns) or stats_df.empty:
        return pd.DataFrame(columns=["hp_id", "score", "t_reach", "stability"])

    df = stats_df.copy()

    # Clean obvious NaNs; keep rows with valid mean & timestep
    df = df.replace([np.inf, -np.inf], np.nan)
    df = df.dropna(subset=["hp_id", "timestep", "mean"])
    if df.empty:
        return pd.DataFrame(columns=["hp_id", "score", "t_reach", "stability"])

    # Ensure types are sensible
    # (hp_id can be any hashable; we'll keep as-is but use a deterministic fallback)
    df = df.sort_values(["hp_id", "timestep"], kind="mergesort")

    grouped = df.groupby("hp_id", sort=False)

    def _tail_slice(g: pd.DataFrame, frac: float) -> pd.DataFrame:
        if len(g) == 0:
            return g
        k = max(1, int(np.ceil(len(g) * max(0.0, min(1.0, frac)))))
        return g.nlargest(k, "timestep")

    def _stability_tail(g: pd.DataFrame, frac: float) -> float:
        # 1 - IQR/|median| on the tail (clipped to [0,1]); 0 if not enough data
        tail = _tail_slice(g, frac)
        if len(tail) < 3:
            return 0.0
        y = tail["mean"].to_numpy(dtype=float)
        median = float(np.median(y))
        if np.isclose(median, 0.0, atol=1e-8):
            # avoid exploding ratio; treat as low stability but defined
            return 0.0
        q25, q75 = np.percentile(y, [25, 75])
        iqr = float(q75 - q25)
        stab = 1.0 - (iqr / (abs(median) + 1e-8))
        return float(np.clip(stab, 0.0, 1.0))

    def _earliest_reach_time(g: pd.DataFrame, target: float, tol: float) -> float:
        """
        Earliest timestep where mean >= target - tol (with linear interpolation).
        Returns +inf if never reached.
        """
        g = g.sort_values("timestep", kind="mergesort")
        t = g["timestep"].to_numpy(dtype=float)
        y = g["mean"].to_numpy(dtype=float)
        if len(t) == 0:
            return float("inf")

        # If first point already at/above target
        if y[0] + tol >= target:
            return float(t[0])

        for i in range(len(t) - 1):
            y0, y1 = y[i], y[i + 1]
            if (y0 + tol) < target <= (y1 + tol):
                dy = y1 - y0
                dt = t[i + 1] - t[i]
                if abs(dy) < 1e-12 or dt <= 0:
                    return float(t[i + 1])
                frac = (target - y0) / dy
                return float(t[i] + frac * dt)

        return float("inf")

    rows = []
    for hp, g in grouped:
        if len(g) == 0:
            continue

        g = g.sort_values("timestep", kind="mergesort")

        if metric == "peak":
            score = float(g["mean"].max())
            # Earliest time to reach (within tol) that peak score
            t_reach = _earliest_reach_time(g, score, tol)
        elif metric == "final":
            tail = _tail_slice(g, tail_frac)
            if tail.empty:
                # fallback to peak if no tail
                score = float(g["mean"].max())
                t_reach = _earliest_reach_time(g, score, tol)
            else:
                score = float(tail["mean"].mean())
                # When using a tail-mean score, use earliest time reaching that score
                t_reach = _earliest_reach_time(g, score, tol)
        else:
            raise ValueError(f"Unsupported metric: {metric}")

        stability = _stability_tail(g, tail_frac)

        # Robust fallbacks for edge cases
        if np.isnan(score):
            # treat NaN score as worst possible
            score = -np.inf
        if not np.isfinite(t_reach):
            # if never reached, put very large time so it loses the tie-break
            t_reach = float("inf")
        if np.isnan(stability):
            stability = 0.0

        rows.append({
            "hp_id": hp,
            "score": score,
            "t_reach": t_reach,
            "stability": stability,
        })

    if not rows:
        return pd.DataFrame(columns=["hp_id", "score", "t_reach", "stability"])

    score_df = pd.DataFrame(rows).sort_values("hp_id", kind="mergesort").reset_index(drop=True)
    return score_df


def select_top_n_hp_ids(
    stats_df: pd.DataFrame,
    n: int,
    metric: str = "peak",         # "peak" or "final"
    tail_frac: float = 0.10,      # used for "final" score and stability
    tol: float = 1e-6,            # tolerance for matching peak/score
) -> list:
    """
    Select top-n hyperparameter IDs with robust, deterministic tie-breaking.

    Hybrid ranking (applies to both metrics):
      1) Higher score wins (peak max or final tail-mean)
      2) If tied: earliest timestep that reaches that score wins
      3) If still tied: higher stability in the tail wins
      4) Final fallback: smaller hp_id (deterministic)

    Args:
        stats_df: Tidy stats for ONE algorithm with columns:
                  ['hp_id', 'timestep', 'mean', ...]
        n: Number of configs to return (<= number available)
        metric: "peak"  -> score = max(mean) over time
                "final" -> score = mean(mean) over last `tail_frac` of timesteps
        tail_frac: Fraction of most-recent points used for "final" score
        tol: Absolute tolerance when checking if a point reaches the score

    Returns:
        List of hp_id (length <= n). Returns [] if input is empty/invalid.
    """
    score_df = compute_hp_ranking_metrics(
        stats_df=stats_df,
        metric=metric,
        tail_frac=tail_frac,
        tol=tol,
    )
    if score_df.empty:
        return []

    # Sort deterministically:
    #   1. score desc
    #   2. t_reach asc
    #   3. stability desc
    #   4. hp_id asc  (final deterministic fallback)
    # Use mergesort for stability.
    score_df = score_df.sort_values(
        by=["score", "t_reach", "stability", "hp_id"],
        ascending=[False, True, False, True],
        kind="mergesort",
    )

    # Return top-n hp_ids
    n = max(1, int(n))
    return score_df["hp_id"].head(n).tolist()



def aggregate_across_hps(
    stats_df: pd.DataFrame,
    milestones: list[int],
    hp_ids: Optional[Iterable] = None,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """
    Aggregate statistics across selected HPs at given milestones using IQM + bootstrap CI.

    Args:
        stats_df: tidy stats for ONE algorithm (columns include: hp_id, timestep, mean)
        milestones: list of evaluation timesteps
        hp_ids: optional iterable of hp_id to include; if None, include all

    Returns:
        (timesteps, iqm, ci_lower, ci_upper)
    """
    if hp_ids is not None:
        stats_df = stats_df[stats_df["hp_id"].isin(list(hp_ids))].copy()

    timesteps, iqm_values, ci_lower_values, ci_upper_values = [], [], [], []

    if stats_df.empty:
        return (np.array([]), np.array([]), np.array([]), np.array([]))

    # Ensure per-hp is time-sorted
    stats_df = stats_df.sort_values(["hp_id", "timestep"])

    for milestone in milestones:
        # for each hp, take the latest <= milestone
        rows = []
        for hp in stats_df["hp_id"].unique():
            g = stats_df[stats_df["hp_id"] == hp]
            g_valid = g[g["timestep"] <= milestone]
            if not g_valid.empty:
                rows.append(float(g_valid.iloc[-1]["mean"]))
            else:
                rows.append(np.nan)

        arr = np.array(rows, dtype=float)
        if np.all(np.isnan(arr)):
            continue

        iqm = compute_iqm(arr)
        lo, hi = bootstrap_ci(arr)

        timesteps.append(milestone)
        iqm_values.append(iqm)
        ci_lower_values.append(lo)
        ci_upper_values.append(hi)

    return (
        np.array(timesteps),
        np.array(iqm_values),
        np.array(ci_lower_values),
        np.array(ci_upper_values),
    )


def aggregate_top_n_across_hps(
    stats_df: pd.DataFrame,
    milestones: list[int],
    top_n: int,
    metric: str = "peak",
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """
    Convenience wrapper: pick top-n hp_ids by metric, then aggregate only those.

    Args:
        stats_df: tidy stats for ONE algorithm
        milestones: eval timesteps
        top_n: number of hyperparameter configs to include
        metric: "peak" or "final"

    Returns:
        (timesteps, iqm, ci_lower, ci_upper)
    """
    hp_ids = select_top_n_hp_ids(stats_df, top_n, metric=metric)
    return aggregate_across_hps(stats_df, milestones, hp_ids=hp_ids)
