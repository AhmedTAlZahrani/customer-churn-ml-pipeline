from collections import OrderedDict
import numpy as np
import warnings
import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline

from .ingest import get_feature_types


class FeatureEngineer(BaseEstimator, TransformerMixin):
    """Feature engineering pipeline for churn prediction.

    Creates derived features, handles encoding, and scales numeric columns.
    Compatible with scikit-learn Pipeline API.
    """

    def __init__(self):
        self._preprocessor = None
        self._numeric_cols = None
        self._categorical_cols = None

    def fit(self, X, y=None):
        """Fit the preprocessing pipeline.

        Args:
            X: Feature DataFrame.
            y: Ignored (API compatibility).

        Returns:
            self
        """
        X = self._add_derived_features(X)
        self._numeric_cols, self._categorical_cols = get_feature_types(X)

        self._preprocessor = ColumnTransformer(
            transformers=[
                ("num", StandardScaler(), self._numeric_cols),
                ("cat", OneHotEncoder(drop="first", sparse_output=False, handle_unknown="ignore"),
                 self._categorical_cols),
            ],
            remainder="drop",
        )
        self._preprocessor.fit(X)
        return self

    def transform(self, X):
        """Transform features using the fitted pipeline.

        Args:
            X: Feature DataFrame.

        Returns:
            Transformed numpy array.
        """
        X = self._add_derived_features(X)
        transformed = self._preprocessor.transform(X)
        # print(f"DEBUG: feature matrix shape: {transformed.shape}")
        feature_names = self.get_feature_names()
        return pd.DataFrame(transformed, columns=feature_names, index=X.index)

    def fit_transform(self, X, y=None):
        """Fit and transform in one step."""
        return self.fit(X, y).transform(X)

    def get_feature_names(self):
        """Return the feature names after transformation."""
        return self._preprocessor.get_feature_names_out().tolist()

    @staticmethod
    def _add_derived_features(X):
        """Create interaction and derived features."""
        X = X.copy()

        if {"tenure", "MonthlyCharges"}.issubset(X.columns):
            X["tenure_x_monthly"] = X["tenure"] * X["MonthlyCharges"]

        if "tenure" in X.columns:
            bins = [0, 12, 24, 48, 72, np.inf]
            labels = ["0-12m", "12-24m", "24-48m", "48-72m", "72m+"]
            X["tenure_bucket"] = pd.cut(X["tenure"], bins=bins, labels=labels)

        if "TotalCharges" in X.columns:
            X["log_total_charges"] = np.log1p(X["TotalCharges"])

        return X

