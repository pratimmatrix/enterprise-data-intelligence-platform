"""
Enterprise Data Loader

Author: Pratim Mistry

Description:
A professional data loader that automatically detects
the file type and loads datasets into a Pandas DataFrame.
"""

from pathlib import Path
import pandas as pd


class DataLoader:
    """
    Enterprise Data Loader

    Supports:
    - CSV (.csv)
    - Excel (.xlsx)

    More formats will be added later:
    - JSON
    - Parquet
    """

    def __init__(self):
        print("DataLoader initialized.")

    def load(self, file_path: str) -> pd.DataFrame:
        """
        Automatically detect file type and load dataset.

        Parameters
        ----------
        file_path : str
            Path of the dataset.

        Returns
        -------
        pd.DataFrame
            Loaded dataset.
        """

        path = Path(file_path)

        if not path.exists():
            raise FileNotFoundError(
                f"File not found: {file_path}"
            )

        suffix = path.suffix.lower()

        if suffix == ".csv":
            return self._load_csv(path)

        elif suffix == ".xlsx":
            return self._load_excel(path)

        else:
            raise ValueError(
                f"Unsupported file type: {suffix}"
            )

    def _load_csv(self, path: Path) -> pd.DataFrame:
        """
        Internal method for loading CSV files.
        """

        print("Loading CSV file...")

        return pd.read_csv(path)

    def _load_excel(self, path: Path) -> pd.DataFrame:
        """
        Internal method for loading Excel files.
        """

        print("Loading Excel file...")

        return pd.read_excel(path)