"""Tests for the data ingestion module."""

import numpy as np
import pandas as pd
import pytest

from src.ingest import load_telco_data, get_feature_types


class TestGetFeatureTypes:
    """Verify numeric/categorical column detection."""

    def test_numeric_columns_detected(self, small_feature_df):
        """Numeric columns should include tenure, MonthlyCharges, etc."""
        numeric, _ = get_feature_types(small_feature_df)
        assert "tenure" in numeric
        assert "MonthlyCharges" in numeric
        assert "TotalCharges" in numeric

    def test_categorical_columns_detected(self, small_feature_df):
        """Object-typed columns should be identified as categorical."""
        _, categorical = get_feature_types(small_feature_df)
        assert "Contract" in categorical
        assert "InternetService" in categorical
        assert "gender" in categorical

    def test_return_types_are_lists(self, small_feature_df):
        """Both return values should be plain lists."""
        numeric, categorical = get_feature_types(small_feature_df)
        assert isinstance(numeric, list)
        assert isinstance(categorical, list)

    def test_no_overlap_between_types(self, small_feature_df):
        """A column should not appear in both numeric and categorical."""
        numeric, categorical = get_feature_types(small_feature_df)
        assert set(numeric).isdisjoint(set(categorical))

    @pytest.mark.parametrize(
        "dtype, expected_group",
        [
            ("int64", "numeric"),
            ("float64", "numeric"),
            ("object", "categorical"),
        ],
    )
    def test_single_column_classification(self, dtype, expected_group):
        """A DataFrame with one column should classify it correctly."""
        df = pd.DataFrame({"col": pd.array([1, 2, 3], dtype=dtype)})
        numeric, categorical = get_feature_types(df)
        if expected_group == "numeric":
            assert "col" in numeric
        else:
            assert "col" in categorical

    def test_empty_dataframe(self):
        """An empty DataFrame should return two empty lists."""
        df = pd.DataFrame()
        numeric, categorical = get_feature_types(df)
        assert numeric == []
        assert categorical == []

    def test_category_dtype_detected(self):
        """Pandas category dtype should appear in the categorical list."""
        df = pd.DataFrame({
            "bucket": pd.Categorical(["a", "b", "c"]),
        })
        _, categorical = get_feature_types(df)
        assert "bucket" in categorical


class TestLoadTelcoData:
    """Test the CSV loading function."""

    def test_missing_file_raises(self, tmp_path):
        """Passing a nonexistent path should raise DataIngestionError."""
        from src.exceptions import DataIngestionError

        bad_path = tmp_path / "nonexistent.csv"
        with pytest.raises(DataIngestionError):
            load_telco_data(str(bad_path))

    def test_loads_from_generated_csv(self, raw_churn_df, tmp_path):
        """Round-trip through CSV should produce valid X, y pair."""
        csv_path = tmp_path / "churn.csv"
        # Convert Churn int back to Yes/No as the loader expects string values
        df = raw_churn_df.copy()
        df["Churn"] = df["Churn"].map({1: "Yes", 0: "No"})
        df.to_csv(csv_path, index=False)

        X, y = load_telco_data(str(csv_path))

        assert "customerID" not in X.columns
        assert "Churn" not in X.columns
        assert set(y.unique()).issubset({0, 1})
        assert len(X) == len(y) == len(raw_churn_df)

    def test_total_charges_coerced_to_numeric(self, raw_churn_df, tmp_path):
        """TotalCharges with whitespace entries should be handled gracefully."""
        df = raw_churn_df.copy()
        df["Churn"] = df["Churn"].map({1: "Yes", 0: "No"})
        # Inject some whitespace values that real Telco data contains
        df.loc[df.index[:3], "TotalCharges"] = " "
        csv_path = tmp_path / "churn_blanks.csv"
        df.to_csv(csv_path, index=False)

        X, y = load_telco_data(str(csv_path))
        assert X["TotalCharges"].isna().sum() == 0
        assert X["TotalCharges"].dtype == np.float64
