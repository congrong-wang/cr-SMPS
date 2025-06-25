from __future__ import annotations

from ..core.smps_data import SMPSData
from ..core.smps_dataset import SMPSDataset
import glob
import numpy as np
import os
import pandas as pd
from zoneinfo import ZoneInfo, ZoneInfoNotFoundError


def _SMPSData_from_csv(
    file_path: str,
    read_metadata: bool = True,
    read_rawdata: bool = False,
    time_zone: str | None = None,
) -> SMPSData:
    """
    Create an instance of SMPSData from a output CSV file.

    This method reads the SMPS CSV file, automatically identifies the valid particle
    size range, and returns a formatted concentration table.

    Each row corresponds to one sampling, and columns are time + valid particle sizes
    (unit: nm). Reading metadata and raw data can be controlled with parameters.

    Parameters
    ----------
    file_path : str
        The path to the SMPS CSV file.
    read_metadata : bool, optional
        Whether to read metadata from the file. Default is True.
    read_rawdata : bool, optional
        Whether to read raw data from the file. Default is False.
    Returns
    -------
    SMPSData
        An instance of SMPSData containing the data and metadata from the file.
    """

    """ Read sample data. """
    inst = SMPSData()
    # Save filename to SMPSData info
    inst.filename = os.path.basename(file_path)  # Keep only the filename, not full path

    # Read data
    df = pd.read_csv(file_path, skiprows=52, low_memory=False)
    # Keep only normal scan rows
    df = df[
        (df["Detector Status"] == "Normal Scan")
        & (df["Classifier Errors"] == "Normal Scan")
    ].copy()

    # Set the "DateTime Sample Start" column as the index
    # Always parse with dayfirst=True to handle dd/mm/yy and dd/mm/yyyy
    """
    Note in some cases, the format might be month-first, but the function is not yet 
    able to handle this. Maybe in the future a parameter can be added to allow 
    specifying month-first parsing.
    """

    def parse_mixed_date(s):
        # Skip NaN values
        if pd.isna(s):
            return pd.NaT
        # Try parsing with multiple formats
        for fmt in (
            "%d/%m/%Y %H:%M:%S",  # e.g., 31/12/2023 23:59:59
            "%d/%m/%y %H:%M:%S",  # e.g., 31/12/23 23:59:59
            "%d/%m/%Y %I:%M:%S %p",  # e.g., 31/12/2023 11:59:59 PM
            "%d/%m/%y %I:%M:%S %p",  # e.g., 31/12/23 11:59:59 PM
        ):
            try:
                return pd.to_datetime(s, format=fmt, dayfirst=True)
            except Exception:
                continue
        # If all formats fail, try parsing without a specific format
        try:
            return pd.to_datetime(s, dayfirst=True)
        except Exception:
            return pd.NaT

    # Apply the parsing function to the "DateTime Sample Start" column
    df["DateTime Sample Start"] = df["DateTime Sample Start"].apply(parse_mixed_date)

    # old version:
    # # Convert "DateTime Sample Start" to pd.datetime64[ns], coercing errors to NaT
    # df["DateTime Sample Start"] = pd.to_datetime(
    #     df["DateTime Sample Start"], errors="coerce", dayfirst=True
    # )

    # Check if the parsing was successful
    if df["DateTime Sample Start"].isna().any():
        # If there are any NaN values, print a warning
        print(
            "Warning: Some DateTime Sample Start entries failed to parse. "
            "Check the date format in the CSV file."
        )
        # print the problematic rows
        bad_rows = df[df["DateTime Sample Start"].isna()]
        print(
            f'Warning: {len(bad_rows)} entries failed to parse "DateTime Sample Start".'
        )
        print("Problematic rows:")
        print(bad_rows)

    # If time_zone is provided, label time zone information to the index.
    # If time_zone is None, the index will remain unknown.
    if time_zone is not None:
        try:
            tz = ZoneInfo(time_zone)
            # Convert pd.datetime64[ns] to timezone-aware datetime64[ns, tz]
            df["DateTime Sample Start"] = df["DateTime Sample Start"].dt.tz_localize(tz)
            inst.time_zone = tz  # add to SMPSData's time_zone attribute
        except ZoneInfoNotFoundError:
            print(f"Time zone '{time_zone}' not found. No time zone will be assigned.")
            print(
                "Find correct time zone names at https://en.wikipedia.org/wiki/List_of_tz_database_time_zones"
            )
            inst.time_zone = None
    else:
        inst.time_zone = None
    # Set the index to "DateTime Sample Start" (type: pd.DatetimeIndex)
    # without time zone: pd.datetime64[ns]
    # with time zone: pd.datetime64[ns, tz]
    df.set_index("DateTime Sample Start", inplace=True)

    # Add start and end time to metadata
    inst.metadata["Start Time"] = df.index.min()
    inst.metadata["End Time"] = df.index.max()
    # Add the number of samples to metadata
    inst.metadata["Number of Samples"] = len(df)

    # Read the 52nd line, this is where "Particle Concentration by Midpoint (nm)" and
    # "Raw Concentration by Midpoint (nm)" are located.
    line52 = pd.read_csv(file_path, skiprows=51, nrows=1, header=None).iloc[0].tolist()
    # Drop column 1, which is the "DateTime Sample Start" column. Since it has already
    # been set as the index in `df`, the length would not match if we keep it.
    line52.pop(1)

    # Get all column names
    cols = df.columns.tolist()
    start_idx = line52.index("Particle Concentration by Midpoint (nm)")
    # Check which kind of raw data is present
    if "Raw Concentration by Midpoint (nm)" in line52:
        end_idx = line52.index("Raw Concentration by Midpoint (nm)")
    elif "Raw Data - Time (s)" in cols:
        end_idx = cols.index("Raw Data - Time (s)") - 1
    else:
        print("New raw data format detected. Please check the file.")
        end_idx = -1  # To the last column
    # Extract all particle size column names based on the identified indices
    diameter_columns = cols[start_idx:end_idx]
    # get dlogDp (the average log width of the particle size bins)
    inst.dlogDp = np.mean(np.diff(np.log10(np.array(diameter_columns, dtype=float))))

    # Automatically determine the actual sampled particle size range: infer from all
    # rows' Lower/Upper Size
    lower_bound = df["Lower Size (nm)"].min()
    upper_bound = df["Upper Size (nm)"].max()
    # Keep only columns actually sampled
    diameter_columns = [
        col for col in diameter_columns if lower_bound <= float(col) <= upper_bound
    ]

    # Return data table containing only valid particle sizes
    inst.sample_data = df[diameter_columns]
    inst.metadata["Lower Size (nm)"] = lower_bound
    inst.metadata["Upper Size (nm)"] = upper_bound

    """ Read sample metadata. """
    inst.sample_metadata = df.iloc[:, 0 : start_idx - 1].copy()

    """ Read metadata unless requested not to. Default is True."""
    if read_metadata:
        # Extract metadata from the first 45 rows
        df = pd.read_csv(file_path, skiprows=0, nrows=45, header=None)
        for index, row in df.iterrows():
            # Do not exclude NaN values, as someone may try to access them, and an error would be raised.
            # It is better to return an empty value than to raise an error.
            inst.metadata[row[0]] = row[1]

    """ Read metadata if requested. Default is False."""
    # First, detect if raw data is present
    # Still under construction
    if read_rawdata:
        print("This feature is still under construction.")

    """" Finally, return the instance. """
    return inst


def _SMPSData_list_from_dir(
    dir_path: str,
    read_metadata: bool = True,
    read_rawdata: bool = False,
    time_zone: str | None = None,
) -> list[SMPSData]:
    """
    Read multiple SMPS CSV files from a folder and return a list of SMPSData instances.

    Parameters
    ----------
    dir_path : str
        The path to the folder containing the SMPS CSV files.
    read_metadata : bool, optional
        Whether to read metadata from the files. Default is True.
    read_rawdata : bool, optional
        Whether to read raw data from the files. Default is False.

    Returns
    -------
    list of SMPSData
        A list of SMPSData instances containing the data and metadata from the files.
    """
    # Ensure the directory path ends with a slash
    if not dir_path.endswith(os.sep):
        dir_path += os.sep
    csv_filenames = glob.glob(f"{dir_path}*SMPS*.csv")
    csv_filenames.sort()  # Sort the filenames for consistent order
    # Print the number of files found
    print(f"Found {len(csv_filenames)} SMPS CSV files in '{dir_path}'")
    for i, fname in enumerate(csv_filenames, 1):
        print(f"    {i}: {os.path.basename(fname)}")
    SMPSData_list = []  # Initialize an empty list to hold SMPSData instances
    # Iterate over each CSV file and read it into an SMPSData instance
    print("Reading...")
    for file_path in csv_filenames:
        smps_file = _SMPSData_from_csv(
            file_path,
            read_metadata=read_metadata,
            read_rawdata=read_rawdata,
            time_zone=time_zone,
        )
        SMPSData_list.append(smps_file)
    print("Done reading.")

    return SMPSData_list


def _SMPSDataset_from_SMPSData_list(SMPSData_list: list[SMPSData]) -> SMPSDataset:
    """
    Create an instance of SMPSDataset from a list of SMPSData instances.
    Parameters
    ----------
    smpsdata_list : list of SMPSData
        A list of SMPSData instances to be included in the dataset.
    Returns
    -------
    SMPSDataset: An instance of SMPSDataset containing the provided SMPSData instances.
    """
    inst = SMPSDataset()
    inst.smpsdata_list = SMPSData_list
    # inst.integrate_data()  # Integrate the data from the SMPSData instances
    # After integration, the instance will have int_sample_data, int_sample_metadata, and int_rawdata populated
    return inst


def _SMPSDataset_from_dir(
    dir_path: str,
    read_metadata: bool = True,
    read_rawdata: bool = False,
    time_zone: str | None = None,
) -> SMPSDataset:
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
    SMPSData_list = _SMPSData_list_from_dir(
        dir_path,
        read_metadata=read_metadata,
        read_rawdata=read_rawdata,
        time_zone=time_zone,
    )
    inst = SMPSDataset()
    inst.smpsdata_list = _SMPSDataset_from_SMPSData_list(SMPSData_list)
    return inst
