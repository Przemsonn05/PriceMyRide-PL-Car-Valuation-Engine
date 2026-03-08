# src/data.py
import pandas as pd

def load_raw_data(file_path):
    """Upload raw CSV file."""
    return pd.read_csv(file_path)

def save_processed_data(df, file_path):
    """Save processed dataframe."""
    df.to_csv(file_path, index=False)