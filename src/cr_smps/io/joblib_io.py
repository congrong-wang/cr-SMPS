import joblib
from ..core.smps_dataset import SMPSDataset


def save_SMPSDataset_to_file(data: SMPSDataset, filename: str, compress: int = 0):
    # use joblib to save the dataset to a file
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
    joblib.dump(data, filename, compress=compress)


def load_SMPSDataset_from_file(filename: str) -> SMPSDataset:
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
    return joblib.load(filename, mmap_mode="r")
