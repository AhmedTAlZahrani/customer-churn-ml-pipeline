from typing import List, Tuple

import pandas as pd
import numpy as np

from .exceptions import DataIngestionError


def load_telco_data(path: str = "data/telco_churn.csv") -> Tuple[pd.DataFrame, pd.Series]:
    """Load and prepare the Telco Customer Churn dataset.

    Handles type conversions, missing values, and target encoding.

    Args:
        path: Path to the CSV file.

    Returns:
        Tuple of (features DataFrame, target Series).

    Raises:
        DataIngestionError: If the file cannot be read or is malformed.
    """
    try:
        df = pd.read_csv(path)
    except FileNotFoundError:
        raise DataIngestionError(
            f"Data file not found: {path}", source_path=str(path)
        )
    except Exception as exc:
        raise DataIngestionError(
            f"Failed to read CSV file '{path}': {exc}", source_path=str(path)
        ) from exc

    if "Churn" not in df.columns:
        raise DataIngestionError(
            "Missing required column 'Churn' in dataset", source_path=str(path)
        )

    df["TotalCharges"] = pd.to_numeric(df["TotalCharges"], errors="coerce")
    df["TotalCharges"].fillna(df["TotalCharges"].median(), inplace=True)

    df["Churn"] = (df["Churn"] == "Yes").astype(int)

    drop_cols = ["customerID"]
    df = df.drop(columns=[c for c in drop_cols if c in df.columns])

    X = df.drop(columns=["Churn"])
    y = df["Churn"]

    print(f"Loaded {len(df)} records | Churn rate: {y.mean():.1%}")
    return X, y


def get_feature_types(X: pd.DataFrame) -> Tuple[List[str], List[str]]:
    """Identify numeric and categorical columns.

    Args:
        X: Feature DataFrame.

    Returns:
        Tuple of (numeric column names, categorical column names).
    """
    numeric = X.select_dtypes(include=[np.number]).columns.tolist()
    categorical = X.select_dtypes(include=["object", "category"]).columns.tolist()
    return numeric, categorical

