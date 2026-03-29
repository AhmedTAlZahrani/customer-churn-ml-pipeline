"""Custom exceptions for the churn prediction pipeline.

Provides granular error types for data ingestion, feature engineering,
and model training failures.
"""


class DataIngestionError(Exception):
    """Raised when data loading or initial preparation fails.

    Args:
        message: Human-readable error description.
        source_path: Path to the file that failed to load.
    """

    def __init__(self, message: str, source_path: str = None):
        self.source_path = source_path
        super().__init__(message)


class FeatureEngineeringError(Exception):
    """Raised when feature transformation or encoding fails.

    Args:
        message: Human-readable error description.
        step: Name of the feature engineering step that failed.
    """

    def __init__(self, message: str, step: str = None):
        self.step = step
        super().__init__(message)


class ModelTrainingError(Exception):
    """Raised when model training or cross-validation fails.

    Args:
        message: Human-readable error description.
        model_name: Name of the model that failed.
    """

    def __init__(self, message: str, model_name: str = None):
        self.model_name = model_name
        super().__init__(message)
