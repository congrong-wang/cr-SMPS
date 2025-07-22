from __future__ import annotations
from dataclasses import dataclass, field
import os

from matplotlib import pyplot as plt
from .smps_data import SMPSData
import pandas as pd

from typing import Optional, Union, Tuple, List
import datetime


@dataclass
class SMPSDataset:
    """
    A class to hold multiple SMPSData instances.
    This class is a structure to hold multiple SMPSData instances, allowing for easy access and manipulation of multiple datasets.
    """

    def __init__(self):
        """
        Initialize an instance of SMPSDataset.
        Parameters
        ----------
        smps_data_list : list of SMPSData
            A list of SMPSData instances to be included in the dataset.
        """
        self.smpsdata_list = []
        self.int_sample_data = pd.DataFrame()
        self.int_sample_metadata = pd.DataFrame()
        self.int_rawdata = pd.DataFrame()

    def __len__(self):
        """
        Return the number of SMPSData instances in the dataset.
        This method allows the use of len() function on SMPSDataset instances.
        Returns
        -------
        int: The number of SMPSData instances in the dataset.
        """
        return len(self.smpsdata_list)

    def __getitem__(self, index):
        """
        Get a SMPSData instance by index.
        This method allows the use of indexing on SMPSDataset instances.
        Parameters
        ----------
        index : int
            The index of the SMPSData instance to retrieve.
        Returns
        -------
        SMPSData: The SMPSData instance at the specified index.
        """
        return self.smpsdata_list[index]

    def sort_by_time(self):
        """
        Sort smpsdata_list by the start time of each SMPSData instance.
        This method sorts the SMPSData instances in the dataset based on their start time.
        Returns
        -------
        None: The method sorts the list in place and does not return anything.
        """
        if len(self) < 2:
            return  # No need to sort if there are less than 2 instances

        self.smpsdata_list.sort(
            key=lambda x: x.metadata["Start Time"]
            # if "Start Time" in x.metadata
            # else pd.Timestamp.min
        )
        # Check if time range overlaps
        for i in range(1, len(self)):
            prev = self.smpsdata_list[i - 1]
            curr = self.smpsdata_list[i]
            prev_end = prev.metadata["End Time"]
            curr_start = curr.metadata["Start Time"]
            if prev_end >= curr_start:
                print(
                    f"WARNING: Time overlap detected between '{prev.filename}' and '{curr.filename}':\n"
                    f"    {prev.filename} ends at {prev_end}\n"
                    f"    {curr.filename} starts at {curr_start}\n"
                )

    def print_time_range(self):
        """
        Print the time range of the integrated sample data.
        This method prints the start and end time of the integrated sample data.
        """
        for i, smps_data in enumerate(self.smpsdata_list, 1):
            print("--------------------------")
            print(f"File {i}: {smps_data.filename}")
            smps_data.print_metadata(["Start Time", "End Time", "Number of Samples"])

    # STOP USING THIS FUNCTION, IT'S NOT WORKING PROPERLY
    # def integrate_data(self):
    #     """
    #     Integrate the sample data from all SMPSData instances in the dataset.
    #     This method checks if all SMPSData instances have the same columns and integrates their sample data into a single DataFrame.
    #     This function should be called after any modification to the `smpsdata_list` attribute.
    #     i.e., when adding or removing SMPSData instances.
    #     Returns: nothing, but updates the instance's `int_sample_data`, `int_sample_metadata`, and `int_rawdata` attributes.
    #     -------
    #     """

    #     # Check number of SMPSData instances first
    #     # If there are no SMPSData instances, print a message and do nothing
    #     if len(self.smpsdata_list) < 1:
    #         print("No SMPSData instances in the dataset. Do nothing.")
    #         return
    #     # If there is only one instance, copy its sample data and metadata directly
    #     elif len(self.smpsdata_list) == 1:
    #         self.int_sample_data = self.smpsdata_list[0].sample_data.copy()
    #         self.int_sample_metadata = self.smpsdata_list[0].sample_metadata.copy()
    #         self.int_rawdata = (
    #             self.smpsdata_list[0].rawdata.copy()
    #             if self.smpsdata_list[0].rawdata is not None
    #             else None
    #         )
    #         return
    #     else:
    #         # Sort the SMPSData instances by time
    #         self.sort_by_time()
    #         # Get the first instance's columns as a reference
    #         reference_columns = self.smpsdata_list[0].sample_data.columns.tolist()
    #         same_columns = True
    #         for smps_data in self.smpsdata_list:
    #             # Check if the columns match the reference columns
    #             if smps_data.sample_data.columns.tolist() != reference_columns:
    #                 same_columns = False
    #                 break
    #         # If all instances have the same columns, proceed with integration
    #         if same_columns:
    #             # Concatenate all sample data into a single DataFrame
    #             self.int_sample_data = pd.concat(
    #                 [smps_data.sample_data for smps_data in self.smpsdata_list],
    #                 ignore_index=False,  # keep the original index
    #             )
    #             # Concatenate all sample metadata into a single DataFrame
    #             self.int_sample_metadata = pd.concat(
    #                 [smps_data.sample_metadata for smps_data in self.smpsdata_list],
    #                 ignore_index=False,
    #             )
    #             # sort the rows by index (time)
    #             # Although files in the list are already sorted by time, there might be time overlap.
    #             self.int_sample_data.sort_index(inplace=True)
    #             self.int_sample_metadata.sort_index(inplace=True)

    #             # Concatenate all raw data into a single DataFrame if available
    #             # Skip this for now, as raw data handling is still under construction

    #         # In the future, we may want to handle cases where columns are not consistent.
    #         # For now, we will just print a message and do nothing.
    #         else:
    #             print(
    #                 "Columns are not consistent across all SMPSData instances, skipping integration."
    #             )
    #             return

    def add_SMPSData(self, smps_data: SMPSData):
        """
        Add a SMPSData instance to the dataset.
        This method adds a SMPSData instance to the dataset and integrates the sample data.
        Parameters
        ----------
        smps_data : SMPSData
            The SMPSData instance to add to the dataset.
        Returns
        -------
        None: The method adds the SMPSData instance to the dataset and does not return anything.
        """
        self.smpsdata_list.append(smps_data)
        # self.integrate_data()  # Uncomment this if you want to integrate data after adding a new instance

    @classmethod
    def read_from_dir(
        cls,
        dir_path: str,
        read_metadata: bool = True,
        read_rawdata: bool = False,
        time_zone: str | None = None,
    ):
        from ..io.reader import _SMPSDataset_from_dir

        """
        Create an instance of SMPSDataset by reading multiple SMPS CSV files from a directory.
        Parameters
        ----------
        dir_path : str
            The path to the directory containing the SMPS CSV files.
        read_metadata : bool, optional
            Whether to read metadata from the files. Default is True.
        read_rawdata : bool, optional
            Whether to read raw data from the files. Default is False.
        Returns
        -------
        SMPSDataset: An instance of SMPSDataset containing the data and metadata from the files.
        """
        return _SMPSDataset_from_dir(
            dir_path,
            read_metadata=read_metadata,
            read_rawdata=read_rawdata,
            time_zone=time_zone,
        )

    def plot_heatmap(
        self,
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
        output_time_zone: Optional[Union[str, datetime.tzinfo]] = None,
    ):
        """
        Plot a heatmap of the sample data in the dataset.
        This method uses the `plot_heatmap` function from the `cr_smps.analysis.plotting` module.
        Parameters
        ----------
        time_range : None, str, tuple, or list, optional
            The time range to plot. If None, plot all data. If a string, plot a single date. If a tuple or list, plot a range.
            Default is None.
        Returns
        -------
        None: The method plots the heatmap and does not return anything.
        """
        from ..analysis.plotting.plot_heatmap import _plot_heatmap

        """ ARGUMENTS FOR PLOTTING """
        fig, ax = plt.subplots(figsize=(24, 8))
        cbar_min = 1e1  # Or adjust based on incoming parameters, will modify later
        cbar_max = 1e4
        # pcm = None  # pcm is the handle for the latest pcolormesh object

        fname, pcm = _plot_heatmap(
            ax=ax,
            dataset=self,
            time_range=time_range,
            output_time_zone=output_time_zone,
        )
        # Set axis labels and title
        ax.set_title("SMPS Particle Size Concentration Heatmap", fontsize=20, y=1.04)

        # Set subtitle based on time_range
        if time_range is not None:
            if isinstance(time_range, str):
                subtitle = f"Date: {time_range}"
            elif isinstance(time_range, (tuple, list)) and len(time_range) == 2:
                start_date = pd.to_datetime(time_range[0]).strftime("%Y-%m-%d")
                end_date = pd.to_datetime(time_range[1]).strftime("%Y-%m-%d")
                if start_date == end_date:
                    subtitle = f"Date: {start_date}"
                else:
                    subtitle = f"Date Range: {start_date} to {end_date}"
            else:
                subtitle = f"Date: {str(time_range)}"
        else:
            # If no time_range specified, get the date range from the dataset
            if hasattr(self, "smpsdata_list") and len(self.smpsdata_list) > 0:
                start_time = min(
                    data.metadata.get("Start Time")
                    for data in self.smpsdata_list
                    if "Start Time" in data.metadata
                )
                end_time = max(
                    data.metadata.get("End Time")
                    for data in self.smpsdata_list
                    if "End Time" in data.metadata
                )
                start_date = pd.to_datetime(start_time).strftime("%Y-%m-%d")
                end_date = pd.to_datetime(end_time).strftime("%Y-%m-%d")
                if start_date == end_date:
                    subtitle = f"Date: {start_date}"
                else:
                    subtitle = f"Date Range: {start_date} to {end_date}"
            else:
                subtitle = "All Available Data"

        # Add the subtitle using ax.text
        ax.text(
            0.5,  # x position in axes coordinates
            1.004,  # y position in axes coordinates
            subtitle,
            fontsize=15,
            ha="center",
            va="bottom",
            transform=ax.transAxes,
        )

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
        # plt.show()
        plt.close(fig)

    def plot_pnsd(self):
        """
        Plot the particle size distribution (PNSD) for each SMPSData instance in the dataset.
        This method uses the `plot_pnsd` function from the `cr_smps.analysis.plotting` module.
        Returns
        -------
        None: The method plots the PNSD and does not return anything.
        """
        from ..analysis.plotting.plot_pnsd import plot_pnsd

        for smps_data in self.smpsdata_list:
            plot_pnsd(smps_data)

    def save_to_file(self, filename):
        from ..io.joblib_io import _save_SMPSDataset_to_file

        """
        Save the SMPSDataset instance to a file using joblib.
        Parameters
        ----------
        filename : str
            The name of the file to save the dataset to.
        Returns
        -------
        None: The method saves the dataset to a file and does not return anything.
        """
        _save_SMPSDataset_to_file(self, filename)

    def load_from_file(self, filename):
        from ..io.joblib_io import _load_SMPSDataset_from_file

        """
        Load a SMPSDataset instance from a file using joblib.
        Parameters
        ----------
        filename : str
            The name of the file to load the dataset from.
        Returns
        -------
        SMPSDataset: An instance of SMPSDataset loaded from the file.
        """
        return _load_SMPSDataset_from_file(filename)
