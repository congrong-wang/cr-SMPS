from __future__ import annotations
from ...core import SMPSDataset
import datetime
from matplotlib.colors import LogNorm
import matplotlib.dates as mdates
import matplotlib.pyplot as plt
import matplotlib.ticker as mtick
import numpy as np
import os
import pandas as pd
from typing import Optional, Union, Tuple, List

# This function will be called by the SMPSDataset class as a method


def _plot_heatmap(
    dataset: SMPSDataset,
    time_range: Optional[
        Union[
            str,
            datetime.datetime,
            pd.Timestamp,
            Tuple[
                Union[str, datetime.datetime, pd.Timestamp],
                Union[str, datetime.datetime, pd.Timestamp],
            ],
            List[Union[str, datetime.datetime, pd.Timestamp]],
            None,
        ]
    ] = None,
    output_dir: Optional[str] = None,
):
    # For now, the problem of this plotting function,
    # is that it does not do averaging for large datasets.
    # Therefore, multiple datapoints would be overlapping each other,
    # and the heatmap would be too dense to read, also lose some information.
    # Need to do automatic averaging according to number of data points
    """
    Generate a heatmap of the sample data for a specific time range.
    This method generates a heatmap of the sample data using matplotlib.
    Parameters
    ----------
    time_range : None, str, tuple, or list, optional
        The time range for which to generate the heatmap. If None, use all data.
        If a string or datetime object, use that specific date.
        If a tuple or list, use the start and end dates as the range.
        Default is None (all data).
    output_dir : str, optional
        The directory where the heatmap image will be saved.
        If None, the image will be saved in the current working directory.
        Default is None.
    Returns
    -------
    None: The method displays the heatmap and does not return anything.
    """
    fig, ax = plt.subplots(figsize=(24, 8))
    cbar_min = 1e1  # Or adjust based on incoming parameters, will modify later
    cbar_max = 1e4
    pcm = None  # pcm is the handle for the latest pcolormesh object

    # Determine the time range for the heatmap
    # If time_range is None, use all data
    if time_range is None:
        xlim = None  # No xlim set, use all data
        # Set the date_str to the start date and the end date, no time
        # Still under construction, so use a placeholder "All Data"
        time_str = "All Data"
        fname = "heatmap_all.png"

        def mask_func(idx):
            return np.full(len(idx), True)  # Select all data

    # If time_range is a single date, use that date
    elif isinstance(
        time_range, (str, pd.Timestamp, datetime.datetime, pd.DateOffset)
    ) or hasattr(time_range, "date"):
        date = pd.to_datetime(time_range).date()
        # Set xlim to the whole day
        start = pd.Timestamp.combine(date, pd.Timestamp.min.time())
        end = pd.Timestamp.combine(date, pd.Timestamp.max.time())
        xlim = (start, end)
        time_str = f"{date.strftime('%Y-%m-%d')}"
        fname = f"heatmap_{date}.png"

        def mask_func(idx):
            return pd.to_datetime(idx).date == date  # All data for that date

    # If time_range is a tuple or list, use that range
    elif isinstance(time_range, (tuple, list)) and len(time_range) == 2:
        start = pd.to_datetime(time_range[0])
        end = pd.to_datetime(time_range[1])
        if start >= end:
            raise ValueError("Start date must be earlier than end date.")
        # Check if only dates are given without time
        if (
            start.hour,
            start.minute,
            start.second,
            start.microsecond,
            end.hour,
            end.minute,
            end.second,
            end.microsecond,
        ) == (0, 0, 0, 0, 0, 0, 0, 0):
            # If only dates are given, set the time to the start and end of the day
            start = pd.Timestamp.combine(start.date(), pd.Timestamp.min.time())
            end = pd.Timestamp.combine(end.date(), pd.Timestamp.max.time())
            time_str = f"{start.date().strftime('%Y-%m-%d')} ~ {end.date().strftime('%Y-%m-%d')}"
            fname = f"heatmap_{start.date().strftime('%Y%m%d')}_{end.date().strftime('%Y%m%d')}.png"
        else:
            time_str = f"{start.strftime('%Y-%m-%d %H:%M:%S')} ~ {end.strftime('%Y-%m-%d %H:%M:%S')}"
            fname = f"heatmap_{start.strftime('%Y%m%d_%H%M%S')}_{end.strftime('%Y%m%d_%H%M%S')}.png"
        xlim = (start, end)

        def mask_func(idx):
            idx_dt = pd.to_datetime(idx)
            return (idx_dt >= start) & (idx_dt <= end)

    else:
        raise ValueError(
            "time_range must be None, a single date, or a (start, end) tuple."
        )

    # Plotting the heatmap
    # Iterate over each SMPSData instance and plot the data
    for s in dataset.smpsdata_list:
        idx = pd.to_datetime(s.sample_data.index)
        mask = mask_func(idx)
        if not mask.any():
            continue  # Skip if no data for this time range
        df = s.sample_data.loc[mask]
        X = df.index
        Y = df.columns.astype(float)  # nm
        Z = df.values.T
        Z[Z < cbar_min] = cbar_min
        pcm = ax.pcolormesh(
            X,
            Y,
            Z,
            shading="nearest",
            norm=LogNorm(vmin=cbar_min, vmax=cbar_max),
            cmap="jet",
        )

    # Only show data within the specified xlim
    if xlim is not None:
        ax.set_xlim(xlim)
    ax.set_ylim([1e1, 1e3])

    """ Set labels and title """
    ax.set_xlabel("Time", fontsize=18)
    ax.set_ylabel("Particle Size (nm)", fontsize=18)
    ax.set_title("SMPS Particle Size Concentration Heatmap", fontsize=20, y=1.04)
    # Add the time string below the title
    ax.text(
        0.5,  # x position in axes coordinates
        1.004,  # y position in axes coordinates
        time_str,
        fontsize=15,
        ha="center",
        va="bottom",
        transform=ax.transAxes,
    )

    """ Tick formatting """
    ax.semilogy()  # Set y-axis to logarithmic scale
    ax.xaxis_date()
    ax.xaxis.set_major_locator(mdates.AutoDateLocator(maxticks=10))
    ax.xaxis.set_major_formatter(mdates.DateFormatter("%Y-%m-%d\n%H:%M"))
    ax.yaxis.set_major_formatter(mtick.ScalarFormatter())

    # Set x-axis tick & labels
    if xlim is not None:
        # Calculate the total number of days in the xlim range,
        # to decide how the x-axis would behave
        total_days = (xlim[1] - xlim[0]).days + 1
        # Within 1 day, use hourly ticks
        if total_days <= 1:
            ax.xaxis.set_major_locator(mdates.HourLocator(interval=1))
            ax.xaxis.set_major_formatter(mdates.DateFormatter("%H"))
            plt.setp(ax.get_xticklabels(), fontsize=16)
        # 1-10 days, use daily major ticks and 06/12/18 hour minor ticks
        elif 1 < total_days <= 12:
            ax.xaxis.set_major_locator(mdates.DayLocator())
            ax.xaxis.set_major_formatter(mdates.DateFormatter("%m-%d"))
            ax.xaxis.set_minor_locator(mdates.HourLocator(byhour=[6, 12, 18]))
            ax.xaxis.set_minor_formatter(mdates.DateFormatter("%H"))
            plt.setp(
                ax.xaxis.get_majorticklabels(),
                fontsize=16,
                va="top",
                y=-0.03,  # vertical location relative to the x-axis
            )
            plt.setp(
                ax.xaxis.get_minorticklabels(),
                fontsize=14,
                va="top",
            )

            # `length` means the length of the tick
            ax.tick_params(axis="x", which="major", length=8)
            ax.tick_params(axis="x", which="minor", length=4)
        else:
            # 超过10天，默认
            ax.xaxis.set_major_locator(mdates.AutoDateLocator(maxticks=10))
            ax.xaxis.set_major_formatter(mdates.DateFormatter("%Y-%m-%d"))
            plt.setp(ax.get_xticklabels(), rotation=30, ha="right", fontsize=14)
    # If no xlim is set, i.e. all data is used, use AutoDateLocator
    else:
        ax.xaxis.set_major_locator(mdates.AutoDateLocator(maxticks=10))
        ax.xaxis.set_major_formatter(mdates.DateFormatter("%Y-%m-%d"))
        plt.setp(ax.get_xticklabels(), rotation=30, ha="right", fontsize=14)

    # Set y-axis tick & labels
    ax.set_ylim([1e1, 1e3])
    ax.set_yticks([1e1, 1e2, 1e3])
    ax.yaxis.set_major_formatter(mtick.LogFormatterMathtext())
    ax.tick_params(axis="y", labelsize=14)

    # Draw colorbar
    if pcm is not None:
        cbar = plt.colorbar(pcm, ax=ax)
        cbar.ax.tick_params(labelsize=16)  # Set colorbar tick label font size
        cbar.set_label(
            "dN/dlogDp [cm$^{-3}$]", fontsize=16
        )  # Set colorbar label font size

    plt.tight_layout()  # `tight_layout` to avoid overlapping labels

    # Save the figure
    if output_dir:
        os.makedirs(output_dir, exist_ok=True)
        save_path = os.path.join(output_dir, fname)
    else:
        save_path = fname
    plt.savefig(save_path, dpi=300, bbox_inches="tight")

    # Show and close
    plt.show()
    plt.close(fig)
