import itertools
from typing import Any

import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns

from hyperlax.hyperparam.distributions import (
    BaseDistribution,
    Categorical,
    DiscreteQuantized,
    LogNormal,
    LogUniform,
    UniformDiscrete,
)


def plot_hyperparam_distributions(
    param_samples: dict[str, np.ndarray],
    distributions: dict[str, BaseDistribution],
    save_path: str = None,
    figsize: tuple = (15, 10),
):
    """
    Visualize the distributions of sampled hyperparameters with a colorblind-friendly palette.

    Args:
        param_samples: Dictionary of parameter names to their sampled values
        distributions: Dictionary of parameter names to their distribution objects
        save_path: Optional path to save the plot
        figsize: Figure size tuple (width, height)
    """
    n_params = len(param_samples)
    n_cols = 3
    n_rows = (n_params + n_cols - 1) // n_cols

    plt.figure(figsize=figsize)
    colors = sns.color_palette("colorblind")

    type_colors = {
        "UniformContinuous": colors[0],
        "LogUniform": colors[1],
        "Normal": colors[2],
        "LogNormal": colors[3],
        "UniformDiscrete": colors[4],
        "DiscreteQuantized": colors[5],
        "Categorical": colors[6],
    }

    for i, (param_name, samples) in enumerate(param_samples.items()):
        ax = plt.subplot(n_rows, n_cols, i + 1)
        dist = distributions[param_name]

        dist_type = dist.__class__.__name__
        color = type_colors.get(dist_type, colors[-1])

        if isinstance(dist, Categorical):
            unique, counts = np.unique(samples, return_counts=True)
            ax.bar(range(len(unique)), counts, alpha=0.7, color=color)
            ax.set_xticks(range(len(unique)))
            ax.set_xticklabels(unique, rotation=45)

        elif isinstance(dist, DiscreteQuantized):
            domain_min, domain_max = dist.domain
            possible_values = np.arange(domain_min, domain_max + dist.scale, dist.scale)
            counts = np.zeros_like(possible_values, dtype=int)

            for idx, val in enumerate(possible_values):
                counts[idx] = np.sum(samples == val)

            ax.bar(possible_values, counts, alpha=0.7, color=color, width=dist.scale * 0.8)

            if domain_max - domain_min > 100:
                step = len(possible_values) // 5
                tick_indices = range(0, len(possible_values), step)
                ax.set_xticks([possible_values[i] for i in tick_indices])
            else:
                ax.set_xticks(possible_values)

            if len(possible_values) > 6:
                plt.setp(ax.get_xticklabels(), rotation=45, ha="right")

        elif isinstance(dist, UniformDiscrete):
            bins = np.arange(dist.domain[0], dist.domain[1] + 2) - 0.5
            ax.hist(samples, bins=bins, alpha=0.7, color=color)
            ax.set_xticks(range(dist.domain[0], dist.domain[1] + 1))

        else:  # Continuous distributions
            if isinstance(dist, (LogUniform, LogNormal)):
                ax.hist(np.log10(samples), bins=20, alpha=0.7, color=color)
                ax.set_xlabel(f"log10({param_name})")

                def sci_format(x, p):
                    return f"1e{int(x)}"

                ax.xaxis.set_major_formatter(plt.FuncFormatter(sci_format))
            else:
                ax.hist(samples, bins=20, alpha=0.7, color=color)

        # Common formatting
        ax.grid(True, alpha=0.3, linestyle="--")
        ax.set_ylabel("Count")
        if not isinstance(dist, (LogUniform, LogNormal)):
            ax.set_xlabel(param_name)

        title = f"{param_name} Distribution\n({dist.__class__.__name__})"
        ax.set_title(title, pad=20)
        title_bbox = dict(facecolor=color, alpha=0.3, edgecolor=color, pad=3.0)
        ax.set_title(title, bbox=title_bbox)

    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, bbox_inches="tight", dpi=300)
        plt.close()
    else:
        plt.show()


def plot_hyperparam_relationships(
    param_samples: dict[str, np.ndarray],
    distributions: dict[str, Any],
    save_path: str = None,
    figsize: tuple = None,
):
    """
    Visualize relationships between all pairs of hyperparameters in a grid.

    Args:
        param_samples: Dictionary of parameter names to their sampled values
        distributions: Dictionary of parameter names to their distribution objects
        save_path: Optional path to save the plot
        figsize: Optional figure size tuple (width, height)
    """
    # Get all parameter names
    param_names = list(param_samples.keys())
    n_params = len(param_names)

    # Calculate figure size if not provided
    if figsize is None:
        # Scale figure size based on number of parameters
        size_per_plot = 2.5
        figsize = (size_per_plot * n_params, size_per_plot * n_params)

    # Create figure and grid of subplots
    fig, axes = plt.subplots(n_params, n_params, figsize=figsize)

    # Create colormap for scatter plots
    colors = sns.color_palette("husl", n_colors=n_params)

    # Iterate through all pairs of parameters
    for i, j in itertools.product(range(n_params), range(n_params)):
        ax = axes[i, j]
        param1 = param_names[j]  # x-axis parameter
        param2 = param_names[i]  # y-axis parameter

        # Get distributions for parameter types
        dist1 = distributions[param1]
        dist2 = distributions[param2]

        # Determine if parameters should be plotted in log scale
        is_log1 = isinstance(dist1, (LogUniform, LogNormal))
        is_log2 = isinstance(dist2, (LogUniform, LogNormal))

        x_data = np.log10(param_samples[param1]) if is_log1 else param_samples[param1]
        y_data = np.log10(param_samples[param2]) if is_log2 else param_samples[param2]

        # Plot diagonal with histograms
        if i == j:
            if isinstance(dist1, Categorical):
                unique, counts = np.unique(x_data, return_counts=True)
                ax.bar(range(len(unique)), counts, alpha=0.7, color=colors[i])
                ax.set_xticks(range(len(dist1.values)))
                ax.set_xticklabels(dist1.values, rotation=45)
            else:
                ax.hist(x_data, bins=20, color=colors[i], alpha=0.7)

            if is_log1:
                ax.set_xlabel(f"log10({param1})")
            else:
                ax.set_xlabel(param1)
        else:
            # Create scatter plot with hexbin for dense regions
            if len(x_data) > 1000:
                ax.hexbin(x_data, y_data, gridsize=20, cmap="YlOrRd")
            else:
                ax.scatter(x_data, y_data, alpha=0.5, s=10, c=[colors[i]], rasterized=True)

            # Handle categorical variables
            if isinstance(dist1, Categorical):
                ax.set_xticks(range(len(dist1.values)))
                ax.set_xticklabels(dist1.values, rotation=45)
            if isinstance(dist2, Categorical):
                ax.set_yticks(range(len(dist2.values)))
                ax.set_yticklabels(dist2.values)

            # Add correlation coefficient for numerical pairs
            if not isinstance(dist1, Categorical) and not isinstance(dist2, Categorical):
                corr = np.corrcoef(x_data, y_data)[0, 1]
                ax.text(
                    0.05,
                    0.95,
                    f"ρ={corr:.2f}",
                    transform=ax.transAxes,
                    verticalalignment="top",
                    fontsize=8,
                )

        # Set labels only for edge plots
        if i == n_params - 1:  # Bottom row
            if is_log1:
                ax.set_xlabel(f"log10({param1})")
            else:
                ax.set_xlabel(param1)
        if j == 0:  # Leftmost column
            if is_log2:
                ax.set_ylabel(f"log10({param2})")
            else:
                ax.set_ylabel(param2)

        # Add grid and style
        ax.grid(True, alpha=0.3, linestyle="--")
        ax.set_facecolor("#f8f9fa")

        # Remove tick labels for interior plots
        if i != n_params - 1:
            ax.set_xticks([])
        if j != 0:
            ax.set_yticks([])

    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, bbox_inches="tight", dpi=300)
        plt.close()
    else:
        plt.show()
