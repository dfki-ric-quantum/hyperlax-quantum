"""Common plotting utilities."""

import logging
from pathlib import Path
from typing import Iterable, Tuple

import matplotlib.pyplot as plt
import plotly.graph_objects as go

logger = logging.getLogger(__name__)


# ---------- Saving ----------

def save_figure(fig, output_dir: Path, filename_base: str):
    """Save figure in multiple formats."""
    output_dir.mkdir(parents=True, exist_ok=True)

    if isinstance(fig, plt.Figure):
        for ext in ['png', 'svg', 'pdf']:
            try:
                fig.savefig(
                    output_dir / f"{filename_base}.{ext}",
                    dpi=300,
                    bbox_inches='tight'
                )
            except Exception as e:
                logger.error(f"Failed to save as {ext}: {e}")
        plt.close(fig)

    elif isinstance(fig, go.Figure):
        # HTML always works
        fig.write_html(str(output_dir / f"{filename_base}.html"))
        # Try static export (requires kaleido)
        try:
            fig.write_image(str(output_dir / f"{filename_base}.webp"))
        except Exception as e:
            logger.warning(
                f"Could not save static image '{filename_base}.webp'. "
                "Install kaleido: pip install kaleido. "
                f"Full Error: {e}"
            )


# ---------- Labels, titles, filenames ----------

# def format_env_name(env: str) -> str:
#     """Clean environment name for display."""
#     return env.replace('_', ' ').replace('-', ' ')
def format_env_name(env: str) -> str:
    return env


# def format_algo_name(algo: str) -> str:
#     """Clean algorithm name for display.
#     Keeps parenthetical variants as-is for display so variants remain distinct.
#     Uppercases base token when no parentheses are present.
#     """
#     return algo if '(' in algo else algo.upper()
def format_algo_name(algo: str) -> str:
    return algo


def format_title(title: str, subtitle: str | None = None, backend: str = "matplotlib") -> str:
    """Uniform title/subtitle formatting for both backends."""
    if not subtitle:
        return title
    if backend == "plotly":
        return f"{title}<br><sub>{subtitle}</sub>"
    return f"{title}\n{subtitle}"


def _sanitize(s: str) -> str:
    return s.replace('.', '_').replace('-', '_').replace(' ', '_')


def get_plot_filename(plot_type: str, env: str, algo: str | None = None, suffix: str = "") -> str:
    """Stable, collision-resistant filename scheme."""
    parts = [plot_type, _sanitize(env)]
    if algo:
        cleaned = (algo.replace('_', '-')
                        .replace(' ', '-')
                        .replace('(', '')
                        .replace(')', ''))
        parts.append(cleaned)
    if suffix:
        parts.append(_sanitize(suffix))
    return "_".join(parts)


def format_hyperparam_name(name: str) -> str:
    """
    Human-friendly formatting for flattened hyperparameter keys.

    Examples:
        'algorithm.hyperparam.learning_rate' -> 'hyperparam › learning rate'
    """
    if not name:
        return name

    clean = str(name).strip()
    if not clean:
        return clean

    # Drop leading namespaces we don't need to repeat.
    prefixes = [
        "algorithm.hyperparam.",
        "algorithm.network.",
        "algorithm.",
        "training.",
        "env_config.",
        "environment.",
    ]
    for prefix in prefixes:
        if clean.startswith(prefix):
            clean = clean[len(prefix):]
            break

    parts = [p for p in clean.split('.') if p]
    pretty_parts: list[str] = []
    for part in parts:
        part = part.replace('_', ' ').replace('-', ' ')
        if not part:
            continue
        pretty_parts.append(part)

    if not pretty_parts:
        return clean.replace('_', ' ')

    return " › ".join(pretty_parts)


# ---------- Family/FA/Variant parsing & presence ----------

def split_family_fa(label: str) -> Tuple[str, str | None, str | None]:
    """
    Parse algorithm label into (family, fa, variant).

    Examples:
      'ppo-mlp (v1)'   -> ('ppo', 'mlp', 'v1')
      'ppo (tuned)'    -> ('ppo', None, 'tuned')
      'sac-drpqc'      -> ('sac', 'drpqc', None)
      'dqn'            -> ('dqn', None, None)
    """
    s = str(label).strip()
    variant = None
    # Extract parenthetical suffix if any
    if '(' in s and s.endswith(')'):
        idx = s.rfind('(')
        if idx != -1:
            variant = s[idx + 1 : -1].strip()
            s = s[:idx].strip()

    parts = s.lower().split('-', 1)
    family = parts[0]
    fa = parts[1] if len(parts) > 1 else None
    return family, fa, variant


def present_families_fas_from_labels(labels: Iterable[str],
                                     family_order: list[str],
                                     fa_order: list[str]) -> Tuple[list[str], list[str]]:
    """Compute ordered present families and FAs from algo labels (variants ignored)."""
    fams, fas = set(), set()
    for lab in labels:
        fam, fa, _ = split_family_fa(lab)
        if fam:
            fams.add(fam)
        if fa:
            fas.add(fa)

    fams = [f for f in family_order if f in fams] + sorted(f for f in fams if f not in family_order)
    fas  = [f for f in fa_order if f in fas] + sorted(f for f in fas if f not in fa_order)
    return fams, fas
