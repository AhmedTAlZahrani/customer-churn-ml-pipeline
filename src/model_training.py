import logging
from pathlib import Path
from typing import Any, Dict, Optional, Union

import joblib
import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier
from lightgbm import LGBMClassifier
from sklearn.model_selection import StratifiedKFold, cross_validate

logger = logging.getLogger(__name__)


class ModelTrainer:
    """Train and compare multiple classification models.

    Supports Logistic Regression, Random Forest, XGBoost, and LightGBM
    with stratified k-fold cross-validation.
    """

    def __init__(self, output_dir: str = "models") -> None:
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.best_model = None
        self.best_model_name = None

    def _get_models(self) -> Dict[str, Any]:
        """Create fresh model instances for training."""
        return {
            "LogisticRegression": LogisticRegression(max_iter=1000, random_state=42),
            "RandomForest": RandomForestClassifier(n_estimators=200, max_depth=10, random_state=42),
            "XGBoost": XGBClassifier(n_estimators=200, max_depth=5, learning_rate=0.1,
                                     eval_metric="logloss", random_state=42),
            "LightGBM": LGBMClassifier(n_estimators=200, max_depth=5, learning_rate=0.1,
                                        random_state=42, verbose=-1),
        }

    def train(self, X: Union[pd.DataFrame, np.ndarray], y: Union[pd.Series, np.ndarray], model_name: str = "XGBoost") -> Any:
        """Train a single model.

        Args:
            X: Feature matrix.
            y: Target vector.
            model_name: Name of the model to train.

        Returns:
            Fitted model instance.
        """
        models = self._get_models()
        model = models[model_name]
        model.fit(X, y)
        return model

    def cross_validate_model(self, X: Union[pd.DataFrame, np.ndarray], y: Union[pd.Series, np.ndarray], model_name: str = "XGBoost", n_folds: int = 5) -> Dict[str, Dict[str, float]]:
        """Run stratified k-fold cross-validation for a model.

        Args:
            X: Feature matrix.
            y: Target vector.
            model_name: Model to evaluate.
            n_folds: Number of CV folds.

        Returns:
            Dict with mean and std for each metric.
        """
        models = self._get_models()
        model = models[model_name]
        cv = StratifiedKFold(n_splits=n_folds, shuffle=True, random_state=42)
        scoring = ["accuracy", "precision", "recall", "f1", "roc_auc"]

        results = cross_validate(model, X, y, cv=cv, scoring=scoring, return_train_score=False)

        return {
            metric: {
                "mean": round(results[f"test_{metric}"].mean(), 4),
                "std": round(results[f"test_{metric}"].std(), 4),
            }
            for metric in scoring
        }

    def benchmark_all(self, X: Union[pd.DataFrame, np.ndarray], y: Union[pd.Series, np.ndarray], n_folds: int = 5) -> pd.DataFrame:
        """Compare all models using cross-validation.

        Args:
            X: Feature matrix.
            y: Target vector.
            n_folds: Number of CV folds.

        Returns:
            DataFrame with comparison results.
        """
        models = self._get_models()
        rows = []
        best_auc = -1

        for name in models:
            logger.info(f"Evaluating {name}...")
            cv_results = self.cross_validate_model(X, y, name, n_folds)
            row = {"Model": name}
            for metric, values in cv_results.items():
                row[metric] = values["mean"]
            rows.append(row)

            if cv_results["roc_auc"]["mean"] > best_auc:
                best_auc = cv_results["roc_auc"]["mean"]
                self.best_model_name = name

        # Train best model on full data
        self.best_model = self.train(X, y, self.best_model_name)
        logger.info(f"Best model: {self.best_model_name} (AUC={best_auc:.4f})")

        return pd.DataFrame(rows).sort_values("roc_auc", ascending=False)

    def save_model(self, model: Optional[Any] = None, name: str = "best_model") -> None:
        """Save model to disk using joblib.

        Args:
            model: Model to save. If None, saves the best model.
            name: Filename (without extension).
        """
        model = model or self.best_model
        path = self.output_dir / f"{name}.pkl"
        joblib.dump(model, path)
        logger.info(f"Model saved to {path}")

    def load_model(self, name: str = "best_model") -> Any:
        """Load model from disk."""
        path = self.output_dir / f"{name}.pkl"
        return joblib.load(path)
