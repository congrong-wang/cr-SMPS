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
from zoneinfo import ZoneInfo, ZoneInfoNotFoundError

# This function will be called by the SMPSDataset class as a method


def _plot_heatmap(
    ax: plt.Axes,
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
    output_time_zone: Optional[Union[str, datetime.tzinfo]] = None,
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
    ax : plt.Axes
        The matplotlib Axes object to plot the heatmap on.
    dataset : SMPSDataset
        The SMPSDataset instance containing the sample data.
        Usually `self` when called as a method of SMPSDataset.
    output_time_zone : Optional[Union[str, datetime.tzinfo]]
        The time zone to use for the x-axis labels and data localization.
        If None, use the time zone of the dataset or the first SMPSData instance.
        If the dataset has no time zone, it will raise an error.
        Default is None.
    time_range : None, str, tuple, or list, optional
        The time range for which to generate the heatmap. If None, use all data.
        If a string or datetime object, use that specific date.
        If a tuple or list, use the start and end dates as the range.
        Default is None (all data).
    Returns
    -------
    fname : str
        The filename of the saved heatmap image.
    pcm : matplotlib.collections.PolyCollection
        The PolyCollection object created by pcolormesh.
        This can be used to add a colorbar or further customize the plot.
    """

    """ ARGUMENTS FOR PLOTTING """
    cbar_min = 1e1  # Or adjust based on incoming parameters, will modify later
    cbar_max = 1e4
    """ CHECK TIME ZONES """

    def check_SMPSDataset_time_zone(dataset):
        """Check the time zones of all SMPSData instances in the dataset.
        Returns
        -------
        int: 2 if all SMPSData instances have the same time zone,
             1 if they have different time zones,
            -1 if all SMPSData instances do not have a time zone,
             0 if some have a time zone and some do not.
        """
        # Gather all time_zone into a list. If time_zone exists, the value will be
        # returned, otherwise None.
        zones = [getattr(s, "time_zone", None) for s in dataset.smpsdata_list]
        all_have_zone = all(z is not None for z in zones)
        if all_have_zone:
            # If all have time zone, check if they are the same
            first_zone = zones[0]
            if all(z == first_zone for z in zones):
                return 2
            else:
                return 1
        all_no_zone = all(z is None for z in zones)
        if all_no_zone:
            return -1
        else:
            return 0

    time_zone_status = check_SMPSDataset_time_zone(dataset)

    if time_zone_status == 0:
        raise ValueError(
            "Some SMPSData instances have a time zone, while others do not. "
            "Please ensure all SMPSData instances have a time zone, or all do not have one."
        )

    # output_time_zone is set
    if output_time_zone is not None:
        if time_zone_status < 0:
            raise ValueError(
                "output_time_zone is set, but all SMPSData instances do not have a time zone. "
                "Please ensure all SMPSData instances have a time zone, or do not set output_time_zone."
            )
    # output_time_zone is NOT set
    else:
        # All SMPSData instances have the same time zone
        if time_zone_status == 2:
            # use that time zone for plotting
            output_time_zone = dataset.smpsdata_list[0].time_zone
        # SMPSData instances have different time zones
        elif time_zone_status == 1:
            raise ValueError(
                "output_time_zone is not set, and SMPSData instances have different time zones. "
                "Could not determine a single time zone for plotting. "
            )

    """ DETERMINE TIME RANGE 
    variables that will be used later:
    xlim: x-axis range
    time_str: the line blow the title, showing the time range
    fname: the filename for saving the heatmap image
    Function mask_func(idx): used to filter data for each SMPSData instance
        idx: pd.DatetimeIndex of the sample data index
    """

    # If time_range is None, use all data
    if time_range is None:
        xlim = None  # No xlim set, use all data
        # Set the date_str to the start date and the end date, no time
        # Still under construction, so use a placeholder "All Data"
        time_str = "All Data"
        fname = "heatmap_all.png"
        xlabel = "Time"

        def mask_func(idx):
            return np.full(len(idx), True)  # Select all data

    # If time_range is a single date, use that date
    elif isinstance(
        time_range, (str, pd.Timestamp, datetime.datetime, pd.DateOffset)
    ) or hasattr(time_range, "date"):
        dt = pd.to_datetime(time_range)  # convert date str to pd.Timestamp

        if output_time_zone is not None:
            # localize the date to the output time zone
            dt = dt.tz_localize(output_time_zone)
        date = dt.date()  # Get the date
        # Set xlim to the whole day
        start = pd.Timestamp.combine(date, pd.Timestamp.min.time())
        end = pd.Timestamp.combine(date, pd.Timestamp.max.time())
        if output_time_zone is not None:
            start = start.tz_localize(output_time_zone)
            end = end.tz_localize(output_time_zone)

        xlim = (start, end)
        time_str = f"{date.strftime('%Y-%m-%d')}"
        fname = f"heatmap_{date}.png"
        if output_time_zone is not None:
            xlabel = "Time ({})".format(output_time_zone)
        else:
            xlabel = "Time (no time zone)"

        def mask_func(idx):
            # must convert idx to output_time_zone, otherwise the comparison will fail
            if output_time_zone is not None:
                if idx.tz is None:
                    idx = idx.tz_localize(output_time_zone)
                else:
                    idx = idx.tz_convert(output_time_zone)

            # .normalize() is used to compare the date only, ignoring the time part
            return idx.normalize() == dt.normalize()

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

        if output_time_zone is not None:
            # localize the start and end to the output time zone
            start = start.tz_localize(output_time_zone)
            end = end.tz_localize(output_time_zone)
            xlabel = "Time ({})".format(output_time_zone)
        xlim = (start, end)
        if output_time_zone is not None:
            xlabel = "Time ({})".format(output_time_zone)
        else:
            xlabel = "Time (no time zone)"

        def mask_func(idx):
            if output_time_zone is not None:
                # If idx is timezone-naive, localize it to the output time zone
                if idx.tz is None:
                    idx = idx.tz_localize(output_time_zone)
                else:
                    # If idx is timezone-aware, convert it to the output time zone
                    idx = idx.tz_convert(output_time_zone)
            idx_dt = pd.to_datetime(idx)
            return (idx_dt >= start) & (idx_dt <= end)

    else:
        raise ValueError(
            "time_range must be None, a single date, or a (start, end) tuple."
        )

    pcm = None  # Initialize pcm to None, will be set later, just in case no data is plotted

    # Plotting the heatmap
    # Iterate over each SMPSData instance and plot the data
    for s in dataset.smpsdata_list:
        idx = s.sample_data.index
        # # debug: to see if the index is timezone-aware
        # if idx.tz is not None:
        #     print(f"SMPSData index is timezone-aware: {idx.tz}")
        # else:
        #     print(f"SMPSData index is NOT timezone-aware.")

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
    if not xlabel:
        xlabel = "time."
    ax.set_xlabel(
        xlabel,  # fontsize=18
    )
    ax.set_ylabel(
        "Particle Size (nm)",  # fontsize=18
    )
    # ax.set_title("SMPS Particle Size Concentration Heatmap", fontsize=20, y=1.04)
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
    ax.xaxis.set_major_formatter(
        mdates.DateFormatter(
            "%Y-%m-%d\n%H:%M",
            tz=ZoneInfo(output_time_zone) if output_time_zone else None,
        )
    )
    ax.yaxis.set_major_formatter(mtick.ScalarFormatter())

    # Set x-axis tick & labels
    if xlim is not None:
        # Calculate the total number of days in the xlim range,
        # to decide how the x-axis would behave
        total_days = (xlim[1] - xlim[0]).days + 1
        # Within 1 day, use hourly ticks
        if total_days <= 1:
            ax.xaxis.set_major_locator(mdates.HourLocator(interval=1))
            ax.xaxis.set_major_formatter(
                mdates.DateFormatter(
                    "%H", tz=ZoneInfo(output_time_zone) if output_time_zone else None
                )
            )
            plt.setp(ax.get_xticklabels(), fontsize=16)
        # 1-10 days, use daily major ticks and 06/12/18 hour minor ticks
        elif 1 < total_days <= 12:
            ax.xaxis.set_major_locator(mdates.DayLocator())
            ax.xaxis.set_major_formatter(
                mdates.DateFormatter(
                    "%m-%d", tz=ZoneInfo(output_time_zone) if output_time_zone else None
                )
            )
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
            ax.xaxis.set_major_formatter(
                mdates.DateFormatter(
                    "%Y-%m-%d",
                    tz=ZoneInfo(output_time_zone) if output_time_zone else None,
                )
            )
            plt.setp(ax.get_xticklabels(), rotation=30, ha="right", fontsize=14)
    # If no xlim is set, i.e. all data is used, use AutoDateLocator
    else:
        ax.xaxis.set_major_locator(mdates.AutoDateLocator(maxticks=10))
        ax.xaxis.set_major_formatter(
            mdates.DateFormatter(
                "%Y-%m-%d", tz=ZoneInfo(output_time_zone) if output_time_zone else None
            )
        )
        plt.setp(ax.get_xticklabels(), rotation=30, ha="right", fontsize=14)

    # Set y-axis tick & labels
    ax.set_ylim([1e1, 1e3])
    ax.set_yticks([1e1, 1e2, 1e3])
    ax.yaxis.set_major_formatter(mtick.LogFormatterMathtext())
    ax.tick_params(axis="y", labelsize=14)

    return fname, pcm
