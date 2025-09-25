# data/loader.py
import pandas as pd
from sklearn.model_selection import train_test_split

def load_data(path, features, target, test_size=0.2, random_state=42):
    """
    Load dataset from CSV, clean it, and return train/test splits.
    """
    df = pd.read_csv(path)

    df = df.dropna()                  # drop missing
    df = df.drop_duplicates()         # remove duplicates

    X = df[features]
    y = df[target]

    return train_test_split(X, y, test_size=test_size, random_state=random_state)
