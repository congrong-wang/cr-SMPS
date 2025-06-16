from __future__ import annotations
from dataclasses import dataclass
import pandas as pd
from typing import Optional, Union, List, Tuple


@dataclass
class SMPSData:
    def __init__(self):
        """
        Initialize an instance of SMPSData.
        This class is a structure to hold data and metadata from a SMPS CSV file.
        Attributes
        ----------
        metadata : dict
            A dictionary to store metadata extracted from the first 45 lines of the file.
        sample_data : pd.DataFrame
            A DataFrame to store the sample data, which includes time and valid particle size concentrations.
        sample_metadata : pd.DataFrame
            A DataFrame to store sample metadata, which corresponds to the sample data.
        rawdata : pd.DataFrame or None
            A DataFrame to store raw data, if applicable. Default is None.
        filename : str or None
            Saves the path to the SMPS CSV file. Default is None.
        """
        self.metadata = {}
        self.sample_data = pd.DataFrame()
        self.sample_metadata = pd.DataFrame()
        self.rawdata = None  # Method to read raw data has not been implemented yet
        self.filename = None  # Original file name
        self.dlogDp = None

    # Return the length of the sample data
    def __len__(self):
        """
        Return the number of samples in the sample data.
        This method allows the use of len() function on SMPSData instances.
        Returns
        -------
        int: The number of samples in the sample data.
        """
        if self.metadata["Number of Samples"]:
            return self.metadata["Number of Samples"]
        else:
            return 0

    def print_data(self):
        """
        Print the data in a readable format.
        """
        if self.sample_data is not None:
            print(self.sample_data.head)
        else:
            print("No data available.")

    def print_metadata(
        self, keys: Optional[Union[str, List[str], Tuple[str, ...]]] = None
    ) -> None:
        # Note here that Tuple[str, ...] any number of strings in a tuple, including 0;
        # where as if written as Tuple[str], it would only accept a single string.
        """
        Print the metadata in a readable format.
        If keys are provided, print only those keys' values.
        Parameters
        ----------
        keys : str or list or tuple, optional
            The key(s) to print from the metadata. If None, print all metadata.
            Default is None.
        Returns
        -------
        None
        """

        # Check if metadata is empty
        if self.metadata is None:
            print("No metadata available.")
        else:
            # If keys are provided, print only those keys' values
            if keys:
                # If a single key is provided as a string
                if isinstance(keys, str):
                    if keys in self.metadata:
                        print(f"{keys}: {self.metadata[keys]}")
                    else:
                        print(f"Metadata '{keys}' is not found.")
                # If multiple keys are provided as a list or tuple of strings
                elif isinstance(
                    keys, (list, tuple) and all(isinstance(k, str) for k in keys)
                ):
                    for k in keys:
                        if k in self.metadata:
                            print(f"{k}: {self.metadata[k]}")
                        else:
                            print(f"Metadata '{k}' is not found.")
            # If no keys are provided, print all metadata
            else:
                for k, v in self.metadata.items():
                    print(f"{k}: {v}")

    def print_columns(self):
        """
        Print the columns of the sample data.
        This method prints the column names of the sample data DataFrame.
        """
        if self.sample_data is not None:
            print("Sample Data Columns:")
            print(self.sample_data.columns)
        else:
            print("No sample data available.")
