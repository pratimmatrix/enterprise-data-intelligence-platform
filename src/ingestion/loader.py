"""
Data Loader Module
==================

This module is responsible for loading datasets into the
Enterprise Data Intelligence Platform.

Author: Pratim Mistry
"""

from pathlib import Path
import pandas as pd


class DataLoader:
    """
    Professional data loading class.

    Supports:
    - CSV
    - Excel (.xlsx)
    """

    def __init__(self):
        print("DataLoader initialized.")