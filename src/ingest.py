import pandas as pd
import numpy as np


def load_telco_data(path="data/telco_churn.csv"):
    """Load and prepare the Telco Customer Churn dataset.

    Handles type conversions, missing values, and target encoding.

    Args:
        path: Path to the CSV file.

    Returns:
        Tuple of (features DataFrame, target Series).
    """
    df = pd.read_csv(path)

    df["TotalCharges"] = pd.to_numeric(df["TotalCharges"], errors="coerce")
    df["TotalCharges"].fillna(df["TotalCharges"].median(), inplace=True)

    df["Churn"] = (df["Churn"] == "Yes").astype(int)

    drop_cols = ["customerID"]
    df = df.drop(columns=[c for c in drop_cols if c in df.columns])

    X = df.drop(columns=["Churn"])
    y = df["Churn"]

    print(f"Loaded {len(df)} records | Churn rate: {y.mean():.1%}")
    return X, y


def get_feature_types(X):
    """Identify numeric and categorical columns.

    Args:
        X: Feature DataFrame.

    Returns:
        Tuple of (numeric column names, categorical column names).
    """
    numeric = X.select_dtypes(include=[np.number]).columns.tolist()
    categorical = X.select_dtypes(include=["object", "category"]).columns.tolist()
    return numeric, categorical

