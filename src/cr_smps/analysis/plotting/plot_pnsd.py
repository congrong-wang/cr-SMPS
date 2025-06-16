from ...core.smps_data import SMPSData
import matplotlib.pyplot as plt


def plot_pnsd(data: SMPSData):
    """
    Each line in the data represents a single scan,
    there might be several scans in a single file.
    I want you to iterate over each scan,
    and plot the particle size concentration for each scan.
    """
    fig, ax = plt.subplots(figsize=(12, 6))
    # Iterate over each SMPSData instance and plot the data
    for idx, row in data.sample_data.iterrows():
        # Plot the particle size concentration for each scan
        ax.plot(
            row.index.astype(float),
            row.values,
            label=idx.strftime("%Y-%m-%d %H:%M:%S"),
            alpha=0.7,
        )
    ax.set_title(data.metadata["Dataset Name"], fontsize=16)
    ax.set_xscale("log")
    ax.set_xlabel("Particle Size (nm)", fontsize=14)
    ax.set_ylabel("Particle Concentration (cm$^{-3}$)", fontsize=14)
    ax.set_ylim([0, 1.8e7])

    ax.legend(loc="upper right", fontsize=10, title="Scan Time")
    ax.grid(True, which="both", linestyle="--", linewidth=0.5)
    plt.tight_layout()
    plt.show()
    # Save the figure with a meaningful name
    fname = f"simple_scan_{data.filename}.png"
    plt.savefig(fname, dpi=300, bbox_inches="tight")

    # close the figure to free up memory
    plt.close(fig)
