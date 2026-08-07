"""
Data Loader Module
==================

Author: Pratim Mistry

Description:
This module is responsible for loading datasets into the
Enterprise Data Intelligence Platform.
"""

from pathlib import Path
import pandas as pd


class DataLoader:
    """
    Enterprise Data Loader

    Supports:
    - CSV
    - Excel (.xlsx)
    """

    def __init__(self):
        print("DataLoader initialized.")

    def load_csv(self, file_path: str) -> pd.DataFrame:
        """
        Load a CSV file.

        Parameters
        ----------
        file_path : str
            Path to the CSV file.

        Returns
        -------
        pd.DataFrame
            Loaded dataset.
        """

        path = Path(file_path)

        if not path.exists():
            raise FileNotFoundError(f"File not found: {file_path}")

        dataframe = pd.read_csv(path)

        return dataframe