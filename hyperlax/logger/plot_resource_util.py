import sys

import matplotlib.dates as mdates
import matplotlib.pyplot as plt
import pandas as pd
from matplotlib.ticker import MaxNLocator


def plot_resource_utilization(csv_file_path):
    # Read the CSV file
    try:
        df = pd.read_csv(csv_file_path)
    except Exception as e:
        print(f"Error reading CSV file: {e}")
        return

    # Convert timestamp to datetime
    df["datetime"] = pd.to_datetime(df["timestamp"], unit="s")

    # Apply a modern style
    plt.style.use("ggplot")

    # Set custom color palette
    colors = {
        "gpu0": "#1f77b4",  # blue
        "gpu1": "#2ca02c",  # green
        "system": "#d62728",  # red
        "process": "#ff7f0e",  # orange
    }

    # Set up the figure with subplots
    fig, axs = plt.subplots(3, 2, figsize=(18, 14), dpi=100)

    # Add a stylized title
    fig.suptitle(
        "GPU Resource Utilization During Training",
        fontsize=22,
        fontweight="bold",
        y=0.98,
        fontfamily="sans-serif",
    )

    # Add a subtle background color to the entire figure
    fig.patch.set_facecolor("#f8f9fa")

    # Plot GPU utilization with enhanced styling
    ax1 = axs[0, 0]
    ax1.plot(
        df["datetime"],
        df["gpu_0_utilization"],
        label="GPU 0",
        color=colors["gpu0"],
        linewidth=2.5,
        alpha=0.85,
    )
    ax1.plot(
        df["datetime"],
        df["gpu_1_utilization"],
        label="GPU 1",
        color=colors["gpu1"],
        linewidth=2.5,
        alpha=0.85,
    )
    ax1.set_title("GPU Utilization (%)", fontsize=16, fontweight="bold", pad=15)
    ax1.set_ylabel("Utilization %", fontsize=14, labelpad=10)
    ax1.set_ylim(0, max(df["gpu_0_utilization"].max(), df["gpu_1_utilization"].max()) * 1.1)
    ax1.yaxis.set_major_locator(MaxNLocator(nbins=10))
    ax1.legend(fontsize=12, frameon=True, facecolor="white", framealpha=0.9)
    ax1.grid(True, linestyle="--", alpha=0.7)
    ax1.spines["top"].set_visible(False)
    ax1.spines["right"].set_visible(False)

    # Plot GPU memory utilization
    ax2 = axs[0, 1]
    ax2.plot(
        df["datetime"],
        df["gpu_0_memory_utilization"],
        label="GPU 0",
        color=colors["gpu0"],
        linewidth=2.5,
        alpha=0.85,
    )
    ax2.plot(
        df["datetime"],
        df["gpu_1_memory_utilization"],
        label="GPU 1",
        color=colors["gpu1"],
        linewidth=2.5,
        alpha=0.85,
    )
    ax2.set_title("GPU Memory Utilization (%)", fontsize=16, fontweight="bold", pad=15)
    ax2.set_ylabel("Memory Utilization %", fontsize=14, labelpad=10)
    ax2.set_ylim(
        0,
        max(df["gpu_0_memory_utilization"].max(), df["gpu_1_memory_utilization"].max()) * 1.1,
    )
    ax2.yaxis.set_major_locator(MaxNLocator(nbins=10))
    ax2.legend(fontsize=12, frameon=True, facecolor="white", framealpha=0.9)
    ax2.grid(True, linestyle="--", alpha=0.7)
    ax2.spines["top"].set_visible(False)
    ax2.spines["right"].set_visible(False)

    # Plot GPU memory used
    ax3 = axs[1, 0]
    ax3.plot(
        df["datetime"],
        df["gpu_0_memory_used_gb"],
        label="GPU 0",
        color=colors["gpu0"],
        linewidth=2.5,
        alpha=0.85,
    )
    ax3.plot(
        df["datetime"],
        df["gpu_1_memory_used_gb"],
        label="GPU 1",
        color=colors["gpu1"],
        linewidth=2.5,
        alpha=0.85,
    )
    ax3.set_title("GPU Memory Used (GB)", fontsize=16, fontweight="bold", pad=15)
    ax3.set_ylabel("Memory (GB)", fontsize=14, labelpad=10)
    ax3.set_ylim(0, max(df["gpu_0_memory_used_gb"].max(), df["gpu_1_memory_used_gb"].max()) * 1.1)
    ax3.yaxis.set_major_locator(MaxNLocator(nbins=10))
    ax3.legend(fontsize=12, frameon=True, facecolor="white", framealpha=0.9)
    ax3.grid(True, linestyle="--", alpha=0.7)
    ax3.spines["top"].set_visible(False)
    ax3.spines["right"].set_visible(False)

    # Plot GPU temperature
    ax4 = axs[1, 1]
    ax4.plot(
        df["datetime"],
        df["gpu_0_temperature"],
        label="GPU 0",
        color=colors["gpu0"],
        linewidth=2.5,
        alpha=0.85,
    )
    ax4.plot(
        df["datetime"],
        df["gpu_1_temperature"],
        label="GPU 1",
        color=colors["gpu1"],
        linewidth=2.5,
        alpha=0.85,
    )
    ax4.set_title("GPU Temperature (°C)", fontsize=16, fontweight="bold", pad=15)
    ax4.set_ylabel("Temperature (°C)", fontsize=14, labelpad=10)
    ax4.set_ylim(
        min(df["gpu_0_temperature"].min(), df["gpu_1_temperature"].min()) * 0.95,
        max(df["gpu_0_temperature"].max(), df["gpu_1_temperature"].max()) * 1.05,
    )
    ax4.yaxis.set_major_locator(MaxNLocator(nbins=10))
    ax4.legend(fontsize=12, frameon=True, facecolor="white", framealpha=0.9)
    ax4.grid(True, linestyle="--", alpha=0.7)
    ax4.spines["top"].set_visible(False)
    ax4.spines["right"].set_visible(False)

    # Plot GPU power usage
    ax5 = axs[2, 0]
    ax5.plot(
        df["datetime"],
        df["gpu_0_power_usage"] / 1000,
        label="GPU 0",
        color=colors["gpu0"],
        linewidth=2.5,
        alpha=0.85,
    )
    ax5.plot(
        df["datetime"],
        df["gpu_1_power_usage"] / 1000,
        label="GPU 1",
        color=colors["gpu1"],
        linewidth=2.5,
        alpha=0.85,
    )
    ax5.set_title("GPU Power Usage (W)", fontsize=16, fontweight="bold", pad=15)
    ax5.set_ylabel("Power (W)", fontsize=14, labelpad=10)
    ax5.yaxis.set_major_locator(MaxNLocator(nbins=10))
    ax5.legend(fontsize=12, frameon=True, facecolor="white", framealpha=0.9)
    ax5.grid(True, linestyle="--", alpha=0.7)
    ax5.spines["top"].set_visible(False)
    ax5.spines["right"].set_visible(False)

    # Plot CPU and Process CPU utilization
    ax6 = axs[2, 1]
    ax6.plot(
        df["datetime"],
        df["system_cpu_percent"],
        label="System CPU",
        color=colors["system"],
        linewidth=2.5,
        alpha=0.85,
    )
    ax6.plot(
        df["datetime"],
        df["process_cpu_percent"],
        label="Process CPU",
        color=colors["process"],
        linewidth=2.5,
        alpha=0.85,
    )
    ax6.set_title("CPU Utilization (%)", fontsize=16, fontweight="bold", pad=15)
    ax6.set_ylabel("Utilization %", fontsize=14, labelpad=10)
    ax6.set_ylim(0, max(df["system_cpu_percent"].max(), df["process_cpu_percent"].max()) * 1.1)
    ax6.yaxis.set_major_locator(MaxNLocator(nbins=10))
    ax6.legend(fontsize=12, frameon=True, facecolor="white", framealpha=0.9)
    ax6.grid(True, linestyle="--", alpha=0.7)
    ax6.spines["top"].set_visible(False)
    ax6.spines["right"].set_visible(False)

    # Format x-axis date formatting
    for ax in axs.flatten():
        ax.set_xlabel("Time", fontsize=14, labelpad=10)
        ax.xaxis.set_major_formatter(mdates.DateFormatter("%H:%M:%S"))
        plt.setp(ax.xaxis.get_majorticklabels(), rotation=45, ha="right")
        ax.tick_params(axis="both", which="major", labelsize=12)
        # Add subtle shading to alternate between plots
        ax.patch.set_facecolor("#f8f8f8")
        ax.patch.set_alpha(0.5)

    # Adjust layout
    plt.tight_layout()
    plt.subplots_adjust(top=0.9, bottom=0.08, left=0.08, right=0.95, hspace=0.55, wspace=0.35)

    # Add a footer with information
    duration = (df["datetime"].max() - df["datetime"].min()).total_seconds() / 60
    fig.text(
        0.5,
        0.01,
        f"Monitoring period: {duration:.1f} minutes • Generated on {pd.Timestamp.now().strftime('%Y-%m-%d')}",
        ha="center",
        fontsize=10,
        style="italic",
        alpha=0.7,
    )

    # Save the plot with high quality
    plt.savefig(
        "gpu_resource_utilization.png",
        dpi=300,
        bbox_inches="tight",
        facecolor=fig.get_facecolor(),
    )

    # Generate statistics
    print("\nResource Utilization Statistics:\n")
    stats = pd.DataFrame(
        {
            "Metric": [
                "GPU 0 Utilization (%)",
                "GPU 1 Utilization (%)",
                "GPU 0 Memory Used (GB)",
                "GPU 1 Memory Used (GB)",
                "GPU 0 Temperature (°C)",
                "GPU 1 Temperature (°C)",
                "GPU 0 Power (W)",
                "GPU 1 Power (W)",
                "System CPU (%)",
                "Process CPU (%)",
            ],
            "Mean": [
                df["gpu_0_utilization"].mean(),
                df["gpu_1_utilization"].mean(),
                df["gpu_0_memory_used_gb"].mean(),
                df["gpu_1_memory_used_gb"].mean(),
                df["gpu_0_temperature"].mean(),
                df["gpu_1_temperature"].mean(),
                df["gpu_0_power_usage"].mean() / 1000,
                df["gpu_1_power_usage"].mean() / 1000,
                df["system_cpu_percent"].mean(),
                df["process_cpu_percent"].mean(),
            ],
            "Max": [
                df["gpu_0_utilization"].max(),
                df["gpu_1_utilization"].max(),
                df["gpu_0_memory_used_gb"].max(),
                df["gpu_1_memory_used_gb"].max(),
                df["gpu_0_temperature"].max(),
                df["gpu_1_temperature"].max(),
                df["gpu_0_power_usage"].max() / 1000,
                df["gpu_1_power_usage"].max() / 1000,
                df["system_cpu_percent"].max(),
                df["process_cpu_percent"].max(),
            ],
            "Min": [
                df["gpu_0_utilization"].min(),
                df["gpu_1_utilization"].min(),
                df["gpu_0_memory_used_gb"].min(),
                df["gpu_1_memory_used_gb"].min(),
                df["gpu_0_temperature"].min(),
                df["gpu_1_temperature"].min(),
                df["gpu_0_power_usage"].min() / 1000,
                df["gpu_1_power_usage"].min() / 1000,
                df["system_cpu_percent"].min(),
                df["process_cpu_percent"].min(),
            ],
        }
    )

    # Format the statistics for better readability
    pd.set_option("display.float_format", "{:.2f}".format)
    print(stats)

    print("\nPlot saved as 'gpu_resource_utilization.png'")


if __name__ == "__main__":
    if len(sys.argv) > 1:
        csv_file_path = sys.argv[1]
        plot_resource_utilization(csv_file_path)
    else:
        print("Please provide the path to the CSV file as an argument.")
        print("Example: python plot_resource_util.py utilization_data.csv")
