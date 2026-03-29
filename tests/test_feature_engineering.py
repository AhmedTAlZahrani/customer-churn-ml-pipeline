"""Tests for the feature engineering module."""

import numpy as np
import pandas as pd
import pytest

from src.feature_engineering import FeatureEngineer


class TestAddDerivedFeatures:
    """Verify that derived / interaction columns are created correctly."""

    def test_tenure_x_monthly_created(self, small_feature_df):
        """tenure_x_monthly should equal tenure * MonthlyCharges."""
        result = FeatureEngineer._add_derived_features(small_feature_df)
        expected = small_feature_df["tenure"] * small_feature_df["MonthlyCharges"]
        pd.testing.assert_series_equal(
            result["tenure_x_monthly"], expected, check_names=False,
        )

    def test_tenure_bucket_created(self, small_feature_df):
        """tenure_bucket should be a categorical column with predefined labels."""
        result = FeatureEngineer._add_derived_features(small_feature_df)
        assert "tenure_bucket" in result.columns
        valid_labels = {"0-12m", "12-24m", "24-48m", "48-72m", "72m+"}
        assert set(result["tenure_bucket"].dropna().unique()).issubset(valid_labels)

    def test_log_total_charges_created(self, small_feature_df):
        """log_total_charges should equal np.log1p(TotalCharges)."""
        result = FeatureEngineer._add_derived_features(small_feature_df)
        expected = np.log1p(small_feature_df["TotalCharges"])
        np.testing.assert_array_almost_equal(
            result["log_total_charges"].values, expected.values,
        )

    def test_original_columns_unchanged(self, small_feature_df):
        """Original DataFrame should not be mutated."""
        original_cols = list(small_feature_df.columns)
        FeatureEngineer._add_derived_features(small_feature_df)
        assert list(small_feature_df.columns) == original_cols

    @pytest.mark.parametrize(
        "tenure_val, expected_bucket",
        [
            (1, "0-12m"),
            (12, "0-12m"),
            (13, "12-24m"),
            (24, "12-24m"),
            (25, "24-48m"),
            (48, "24-48m"),
            (49, "48-72m"),
            (72, "48-72m"),
            (73, "72m+"),
        ],
    )
    def test_tenure_bucket_boundaries(self, tenure_val, expected_bucket):
        """Bucket boundaries should assign edge values correctly.

        pd.cut uses half-open (left, right] intervals, so exact boundary
        values belong to the interval whose right edge they match.
        """
        df = pd.DataFrame({
            "tenure": [tenure_val],
            "MonthlyCharges": [50.0],
            "TotalCharges": [500.0],
        })
        result = FeatureEngineer._add_derived_features(df)
        assert str(result["tenure_bucket"].iloc[0]) == expected_bucket


class TestFeatureEngineerFitTransform:
    """Test the full fit/transform lifecycle of FeatureEngineer."""

    def test_fit_returns_self(self, small_feature_df):
        """fit() should return the FeatureEngineer instance."""
        fe = FeatureEngineer()
        result = fe.fit(small_feature_df)
        assert result is fe

    def test_transform_returns_dataframe(self, small_feature_df):
        """transform() should produce a DataFrame."""
        fe = FeatureEngineer()
        fe.fit(small_feature_df)
        transformed = fe.transform(small_feature_df)
        assert isinstance(transformed, pd.DataFrame)

    def test_output_row_count_matches_input(self, small_feature_df):
        """Row count should be preserved after transformation."""
        fe = FeatureEngineer()
        transformed = fe.fit_transform(small_feature_df)
        assert len(transformed) == len(small_feature_df)

    def test_output_has_no_nans(self, feature_target_pair):
        """Transformed output should be free of NaN values."""
        X, _ = feature_target_pair
        fe = FeatureEngineer()
        transformed = fe.fit_transform(X)
        assert transformed.isna().sum().sum() == 0

    def test_get_feature_names_length(self, small_feature_df):
        """get_feature_names should return one name per output column."""
        fe = FeatureEngineer()
        fe.fit(small_feature_df)
        transformed = fe.transform(small_feature_df)
        names = fe.get_feature_names()
        assert len(names) == transformed.shape[1]

    def test_fit_transform_matches_separate_calls(self, small_feature_df):
        """fit_transform should produce the same result as fit then transform."""
        fe1 = FeatureEngineer()
        combined = fe1.fit_transform(small_feature_df)

        fe2 = FeatureEngineer()
        fe2.fit(small_feature_df)
        separate = fe2.transform(small_feature_df)

        pd.testing.assert_frame_equal(combined, separate)

    def test_index_preserved(self, small_feature_df):
        """DataFrame index from input should carry through to output."""
        df = small_feature_df.copy()
        df.index = [10, 20, 30, 40, 50]
        fe = FeatureEngineer()
        transformed = fe.fit_transform(df)
        assert list(transformed.index) == [10, 20, 30, 40, 50]

    @pytest.mark.parametrize("n_records", [50, 200, 500])
    def test_various_dataset_sizes(self, n_records):
        """FeatureEngineer should handle different dataset sizes."""
        from src.data_generator import generate_churn_data

        df = generate_churn_data(n_records=n_records, random_state=7)
        df = df.drop(columns=["customerID", "Churn"])
        fe = FeatureEngineer()
        transformed = fe.fit_transform(df)
        assert len(transformed) == n_records
        assert transformed.shape[1] > 0
