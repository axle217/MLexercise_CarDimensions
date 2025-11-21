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
                # .str.replace('/', '', regex=False)
        )
    return df

def specific_data_operation(df, source_col='Weight Distribution', target_cols=('Front Weight %', 'Rear Weight %')):
    """
    Splits a column with 'front/rear' weight distribution into two float columns.
    
    Parameters:
        df (pd.DataFrame): input dataframe
        source_col (str): column to split
        target_cols (tuple): names of the two resulting columns
    
    Returns:
        pd.DataFrame: df with two new columns added
    """
    if source_col not in df.columns:
        raise ValueError(f"Column '{source_col}' not found in DataFrame")
    
    # Split column safely
    split_df = df[source_col].astype(str).str.split('/', expand=True)
    
    # Make sure we have exactly 2 columns
    if split_df.shape[1] != 2:
        raise ValueError(f"Column '{source_col}' does not have exactly 2 parts after split")
    
    # Convert to float
    split_df = split_df.astype(float)
    
    # Assign to target columns
    df[list(target_cols)] = split_df

    return df

def data_cleaning(path, rename_columns: dict = None, cleaned_data_path="data/cleaned_data.csv"):
    """
    Load dataset from CSV, clean it, and save it.
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
    merged_data = specific_data_operation(merged_data)
    merged_data = auto_fix_mixed_columns(merged_data)
    merged_data = auto_clean_string_columns(merged_data)
    merged_data = merged_data.drop_duplicates()
    
    return merged_data.to_csv(cleaned_data_path, index=False)

def load_cleaned_data(path, features, target, test_size=0.2, random_state=42):
    """
    Load cleaned dataset from CSV and return train/test splits.
    """
    df = pd.read_csv(path)

    X = df[features]
    y = df[target]

    return train_test_split(X, y, test_size=test_size, random_state=random_state)