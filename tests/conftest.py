"""Shared pytest fixtures for customer-churn-ml-pipeline tests."""

import numpy as np
import pandas as pd
import pytest

from src.data_generator import generate_churn_data


@pytest.fixture()
def raw_churn_df():
    """Full synthetic churn DataFrame with customerID and Churn columns.

    Returns a 200-row DataFrame that mirrors the Telco Churn schema produced
    by ``data_generator.generate_churn_data``.
    """
    return generate_churn_data(n_records=200, random_state=99)


@pytest.fixture()
def feature_target_pair(raw_churn_df):
    """Prepared (X, y) split ready for feature engineering and training.

    Drops ``customerID`` and separates the ``Churn`` target, matching the
    preprocessing that ``ingest.load_telco_data`` performs.
    """
    df = raw_churn_df.drop(columns=["customerID"])
    X = df.drop(columns=["Churn"])
    y = df["Churn"]
    return X, y


@pytest.fixture()
def small_feature_df():
    """Tiny 5-row feature DataFrame for lightweight unit tests.

    Contains the minimum columns required by ``FeatureEngineer._add_derived_features``.
    """
    return pd.DataFrame({
        "tenure": [1, 15, 30, 55, 70],
        "MonthlyCharges": [29.85, 56.95, 53.85, 108.15, 20.00],
        "TotalCharges": [29.85, 854.25, 1615.50, 5953.25, 1400.00],
        "Contract": [
            "Month-to-month",
            "One year",
            "Two year",
            "Month-to-month",
            "One year",
        ],
        "InternetService": ["DSL", "Fiber optic", "No", "Fiber optic", "DSL"],
        "PaymentMethod": [
            "Electronic check",
            "Mailed check",
            "Bank transfer (automatic)",
            "Credit card (automatic)",
            "Mailed check",
        ],
        "gender": ["Male", "Female", "Female", "Male", "Female"],
        "SeniorCitizen": [0, 0, 1, 1, 0],
        "Partner": ["Yes", "No", "Yes", "No", "Yes"],
        "Dependents": ["No", "No", "Yes", "No", "Yes"],
        "PhoneService": ["Yes", "Yes", "Yes", "No", "Yes"],
        "OnlineSecurity": ["No", "Yes", "No internet service", "No", "Yes"],
        "TechSupport": ["No", "No", "No internet service", "Yes", "No"],
        "StreamingTV": ["No", "Yes", "No internet service", "Yes", "No"],
    })


@pytest.fixture()
def binary_target():
    """Binary target vector matching ``small_feature_df`` length."""
    return pd.Series([1, 0, 0, 1, 0], name="Churn")
