"""Compute performance metrics from episode data."""

import logging
from dataclasses import dataclass
from typing import Optional

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)

@dataclass
class SummaryMetrics:
    """Performance summary for a single hyperparameter configuration."""

    hp_id: int
    peak_return: float
    peak_timestep: int
    final_return: float
    auc_at_budget: float
    stability_score: float
    time_to_target: float

    # Learning curve features
    early_learning_rate: float
    forgetting_score: float

    def to_dict(self) -> dict:
        return {
            'hp_id': self.hp_id,
            'peak_return': self.peak_return,
            'peak_timestep': self.peak_timestep,
            'final_return': self.final_return,
            'auc_at_budget': self.auc_at_budget,
            'stability_score': self.stability_score,
            'time_to_target': self.time_to_target,
            'early_learning_rate': self.early_learning_rate,
            'forgetting_score': self.forgetting_score,
        }


def compute_auc(
    timesteps: np.ndarray,
    returns: np.ndarray,
    budget: Optional[float] = None,
    target: Optional[float] = None,
) -> float:
    """
    Compute normalized AUC (Area Under Curve) ∈ [0, 1].

    Normalization: (return - baseline) / (target - baseline)
    where baseline = min(returns) and target = provided or 90th percentile.
    """
    if len(timesteps) == 0 or np.all(np.isnan(returns)):
        return 0.0

    # Sort by time
    sorted_idx = np.argsort(timesteps)
    t = timesteps[sorted_idx]
    y = returns[sorted_idx]

    # Clip to budget
    if budget is not None:
        mask = t <= budget
        t = t[mask]
        y = y[mask]

    if len(t) < 2:
        return 0.0

    # Baseline and target
    baseline = float(np.min(y))
    if target is None or target <= baseline:
        target = float(np.percentile(y, 90))
    target = max(target, baseline + 1e-6)  # Ensure target > baseline

    # Normalize
    y_norm = np.clip((y - baseline) / (target - baseline), 0, 1)

    # Trapezoidal integration
    auc = np.trapezoid(y_norm, t)
    duration = t[-1] - t[0]

    return float(auc / duration) if duration > 0 else 0.0


def compute_time_to_target(
    timesteps: np.ndarray,
    returns: np.ndarray,
    target: float,
    budget: Optional[float] = None,
) -> float:
    """
    Find first timestep where return crosses target (with linear interpolation).

    Returns:
        Timestep of crossing, or budget/inf if never reached.
    """
    if len(timesteps) == 0 or target is None:
        return float('inf')

    # Sort and filter
    sorted_idx = np.argsort(timesteps)
    t = timesteps[sorted_idx]
    y = returns[sorted_idx]

    if budget is not None:
        mask = t <= budget
        t = t[mask]
        y = y[mask]

    if len(t) == 0:
        return float('inf')

    # Already at target at start
    if y[0] >= target:
        return float(t[0])

    # Find crossing
    for i in range(len(t) - 1):
        if y[i] < target <= y[i + 1]:
            # Linear interpolation
            if abs(y[i + 1] - y[i]) < 1e-12:
                return float(t[i + 1])
            frac = (target - y[i]) / (y[i + 1] - y[i])
            t_cross = t[i] + frac * (t[i + 1] - t[i])
            return float(t_cross)

    # Never reached
    return float(budget) if budget else float(t[-1])


def compute_stability(stats_df: pd.DataFrame, tail_frac: float = 0.1) -> float:
    """
    Compute stability as 1 - (IQR / |median|) over the last tail_frac of run.

    Higher = more stable. Returns 0 if insufficient data.
    """
    if len(stats_df) < 3:
        return 0.0

    # Take last 10% of timesteps
    tail_n = max(3, int(len(stats_df) * tail_frac))
    tail = stats_df.nlargest(tail_n, 'timestep')

    if len(tail) < 2:
        return 0.0

    returns = tail['mean'].values
    median = np.median(returns)
    q25 = np.percentile(returns, 25)
    q75 = np.percentile(returns, 75)
    iqr = q75 - q25

    stability = 1.0 - (iqr / (abs(median) + 1e-8))
    return float(np.clip(stability, 0, 1))


def compute_learning_rate(stats_df: pd.DataFrame) -> float:
    """Compute early learning rate (improvement / time) in first half of run."""
    if len(stats_df) < 2:
        return 0.0

    mid_idx = len(stats_df) // 2
    first_half = stats_df.nsmallest(mid_idx, 'timestep')

    if len(first_half) < 2:
        return 0.0

    first_return = first_half.iloc[0]['mean']
    mid_return = first_half.iloc[-1]['mean']
    first_time = first_half.iloc[0]['timestep']
    mid_time = first_half.iloc[-1]['timestep']

    dt = mid_time - first_time
    return float((mid_return - first_return) / dt) if dt > 0 else 0.0


def compute_forgetting_score(stats_df: pd.DataFrame) -> float:
    """
    Forgetting score = (peak - final) / peak ∈ [0, 1].

    0 = no forgetting, 1 = complete forgetting.
    """
    if len(stats_df) < 3:
        return 1.0  # Worst case

    peak_return = stats_df['mean'].max()

    # Final = average of last 10%
    tail_n = max(1, int(len(stats_df) * 0.1))
    final_return = stats_df.nlargest(tail_n, 'timestep')['mean'].mean()

    if abs(peak_return) < 1e-8:
        return 0.0

    forgetting = (peak_return - final_return) / abs(peak_return)
    return float(np.clip(forgetting, 0, 1))


def compute_summary_metrics(
    stats_df: pd.DataFrame,
    budget: Optional[float] = None,
    target: Optional[float] = None,
) -> SummaryMetrics:
    """
    Compute all summary metrics for a single HP configuration.

    Args:
        stats_df: DataFrame with columns [timestep, mean, std, ...]
        budget: Budget timesteps for AUC calculation
        target: Target return for time-to-target
    """
    if len(stats_df) == 0:
        return SummaryMetrics(
            hp_id=-1,
            peak_return=float('-inf'),
            peak_timestep=0,
            final_return=float('-inf'),
            auc_at_budget=0.0,
            stability_score=0.0,
            time_to_target=float('inf'),
            early_learning_rate=0.0,
            forgetting_score=1.0,
        )

    hp_id = int(stats_df.iloc[0]['hp_id'])

    # Peak
    peak_idx = stats_df['mean'].idxmax()
    peak_return = float(stats_df.loc[peak_idx, 'mean'])
    peak_timestep = int(stats_df.loc[peak_idx, 'timestep'])

    # Final (last 10%)
    tail_n = max(1, int(len(stats_df) * 0.1))
    final_return = float(stats_df.nlargest(tail_n, 'timestep')['mean'].mean())

    # Time series
    timesteps = stats_df['timestep'].values
    returns = stats_df['mean'].values

    # AUC
    auc = compute_auc(timesteps, returns, budget, target)

    # Time to target (use 90% of peak if no target provided)
    eff_target = target if target is not None else 0.9 * peak_return
    t_at_target = compute_time_to_target(timesteps, returns, eff_target, budget)

    # Stability and learning
    stability = compute_stability(stats_df)
    learning_rate = compute_learning_rate(stats_df)
    forgetting = compute_forgetting_score(stats_df)

    return SummaryMetrics(
        hp_id=hp_id,
        peak_return=peak_return,
        peak_timestep=peak_timestep,
        final_return=final_return,
        auc_at_budget=auc,
        stability_score=stability,
        time_to_target=t_at_target,
        early_learning_rate=learning_rate,
        forgetting_score=forgetting,
    )



def compute_all_summaries(
    stats_df: pd.DataFrame,
    budget: Optional[float] = None,
    target: Optional[float] = None,
) -> list["SummaryMetrics"]:
    """
    Compute summary metrics for each unique hp_id in the given tidy stats DataFrame.
    Prints simple progress information and includes configs that resulted in all NaNs.
    """
    summaries = []
    hp_ids = sorted(stats_df["hp_id"].unique())
    total = len(hp_ids)

    for i, hp_id in enumerate(hp_ids, start=1):

        hp_stats = stats_df[stats_df["hp_id"] == hp_id]

        # Handle configs with no valid mean values
        if not hp_stats["mean"].notna().any():
            logger.warn(f"     Recording all-NaN config for hp_id={hp_id}.")
            nan_summary = SummaryMetrics(
                hp_id=int(hp_id),
                peak_return=np.nan,
                peak_timestep=0,
                final_return=np.nan,
                auc_at_budget=0.0,  # compute_auc returns 0 for all-NaN
                stability_score=np.nan,
                time_to_target=float('inf'), # compute_time_to_target returns inf
                early_learning_rate=np.nan,
                forgetting_score=np.nan, # results in NaN from (nan-nan)/nan
            )
            summaries.append(nan_summary)
            continue

        try:
            summary = compute_summary_metrics(hp_stats, budget, target)
            summaries.append(summary)
        except Exception as e:
            logger.error(f"     Error on hp_id={hp_id}: {e}. Skipping (valid but errored).")
            continue

    return summaries
