# data/loader.py
import pandas as pd
from sklearn.model_selection import train_test_split
import glob
import re
import numpy as np

def auto_fix_mixed_columns(df, threshold=0.7):
    for col in df.columns:
        # Skip columns already numeric
        if pd.api.types.is_numeric_dtype(df[col]):
            continue

        # Try to interpret strings as numbers
        cleaned = pd.to_numeric(df[col], errors="coerce")

        # Count numeric-like values (not NaN after conversion)
        numeric_fraction = cleaned.notna().mean()

        # Convert only if the column is mostly numeric
        if numeric_fraction >= threshold:
            df[col] = cleaned  # convert to float/int
        else:
            # Leave the column unchanged (normal string column)
            df[col] = df[col].astype(str)

    return df

def auto_clean_string_columns(df):
    string_cols = [
        col for col in df.columns
        if df[col].apply(lambda x: isinstance(x, str) or pd.isna(x)).all()
    ]

    for col in string_cols:
        df[col] = (
            df[col].astype(str)
                .str.upper()
                .str.replace('\xa0', '', regex=False)
                .str.replace(' ', '', regex=False)
                .str.replace('-', '', regex=False)
                .str.replace('.', '', regex=False)
                .str.replace('/', '', regex=False)
        )
    return df


def clean_and_load_data(path, features, target, test_size=0.2, random_state=42, rename_columns: dict = None, save_cleaned_data=False, cleaned_data_path="data/cleaned_data.csv"):
    """
    Load dataset from CSV, clean it, and return train/test splits.
    """
    # if path is a folder:  path = "data/"
    # if path is a pattern: path = "data/*.csv"
    files = glob.glob(path)   # <-- expands to a list of CSV files
    # print(files)

    if len(files) == 0:
        raise FileNotFoundError(f"No files found at: {path}")

    dataframes = [pd.read_csv(f, encoding='ISO-8859-1') for f in files]

    # dataframes = [pd.read_csv(files, encoding='ISO-8859-1') for files in path]
    removed_NaN_dataframes = [df.dropna() for df in dataframes]

    merged_data = pd.concat(removed_NaN_dataframes)
    # Remove non-breaking white spaces in string type entries
    merged_data = merged_data.map(
        lambda x: x.strip().replace('\xa0', '') if isinstance(x, str) else x
    )

    # Check for white spaces in label
    # print(merged_data.columns.tolist())
    merged_data.columns = (
        merged_data.columns
        .str.strip()  # Remove leading/trailing whitespace
        .str.replace('\xa0', '', regex=False)  # Remove non-breaking spaces
        # .str.replace(' ', '_')  # Replace spaces with underscores
    )
    # print(merged_data.columns.tolist())

    if rename_columns:
        merged_data = merged_data.rename(columns=rename_columns)

    # print(merged_data.columns.tolist())

    # Cleaning typos and duplicates
    merged_data = auto_fix_mixed_columns(merged_data)
    merged_data = auto_clean_string_columns(merged_data)
    merged_data = merged_data.drop_duplicates()
    
    # Save cleaned data to CSV
    if save_cleaned_data:
        merged_data.to_csv(cleaned_data_path, index=False)

    X = merged_data[features]
    y = merged_data[target]

    return train_test_split(X, y, test_size=test_size, random_state=random_state)


def load_cleaned_data(path, features, target, test_size=0.2, random_state=42):
    """
    Load cleaned dataset from CSV and return train/test splits.
    """
    df = pd.read_csv(path)

    X = df[features]
    y = df[target]

    return train_test_split(X, y, test_size=test_size, random_state=random_state)