# src/data.py

import pandas as pd
from pathlib import Path
from typing import Union


def load_raw_data(file_path: Union[str, Path]) -> pd.DataFrame:
    """
    Load raw CSV file.
    
    Parameters
    ----------
    file_path : str or Path
        Path to CSV file
        
    Returns
    -------
    pd.DataFrame
        Raw data
        
    Raises
    ------
    FileNotFoundError
        If file doesn't exist
    """
    file_path = Path(file_path)
    
    if not file_path.exists():
        raise FileNotFoundError(f"File not found: {file_path}")
    
    return pd.read_csv(file_path)


def save_processed_data(df: pd.DataFrame, file_path: Union[str, Path]) -> None:
    """
    Save processed DataFrame to CSV.
    
    Parameters
    ----------
    df : pd.DataFrame
        DataFrame to save
    file_path : str or Path
        Destination path
    """
    file_path = Path(file_path)
    file_path.parent.mkdir(parents=True, exist_ok=True)
    
    df.to_csv(file_path, index=False)
    print(f"✓ Saved to: {file_path}")