from typing import Any, Dict, List, Optional, Union

import numpy as np
import pandas as pd
import shap


class ModelExplainer:
    """SHAP-based model explainability for churn predictions.

    Provides global feature importance and local instance-level explanations.
    """

    def __init__(self, model: Any, feature_names: Optional[List[str]] = None) -> None:
        self.model = model
        self.feature_names = feature_names
        self._explainer = None

    def _get_explainer(self, X_background: pd.DataFrame) -> shap.Explainer:
        """Create a SHAP explainer with background data."""
        if self._explainer is None:
            if hasattr(self.model, "predict_proba"):
                self._explainer = shap.Explainer(self.model, X_background)
            else:
                self._explainer = shap.TreeExplainer(self.model)
        return self._explainer

    def explain_global(self, X: pd.DataFrame, max_samples: int = 500) -> pd.DataFrame:
        """Compute global SHAP feature importance.

        Args:
            X: Feature matrix (DataFrame or array).
            max_samples: Max samples for background data.

        Returns:
            DataFrame with features sorted by mean |SHAP value|.
        """
        background = X.sample(n=min(max_samples, len(X)), random_state=42)
        explainer = self._get_explainer(background)
        shap_values = explainer(X)

        if isinstance(shap_values.values, list):
            vals = np.abs(shap_values.values[1]).mean(axis=0)
        else:
            vals = np.abs(shap_values.values).mean(axis=0)

        names = self.feature_names or [f"f{i}" for i in range(len(vals))]
        importance = pd.DataFrame({
            "feature": names[:len(vals)],
            "mean_shap": vals,
        }).sort_values("mean_shap", ascending=False).reset_index(drop=True)

        return importance

    def explain_instance(self, X: pd.DataFrame, idx: int = 0, max_samples: int = 200) -> Dict[str, Any]:
        """Explain a single prediction.

        Args:
            X: Feature matrix.
            idx: Row index to explain.
            max_samples: Max background samples.

        Returns:
            Dict with prediction, feature contributions.
        """
        background = X.sample(n=min(max_samples, len(X)), random_state=42)
        explainer = self._get_explainer(background)
        row = X.iloc[[idx]]
        shap_values = explainer(row)

        if isinstance(shap_values.values, list):
            vals = shap_values.values[1][0]
        else:
            vals = shap_values.values[0]

        names = self.feature_names or [f"f{i}" for i in range(len(vals))]
        contributions = sorted(
            zip(names[:len(vals)], vals),
            key=lambda x: abs(x[1]),
            reverse=True,
        )

        prediction = self.model.predict_proba(row)[0] if hasattr(self.model, "predict_proba") else None

        return {
            "index": idx,
            "churn_probability": round(float(prediction[1]), 4) if prediction is not None else None,
            "top_factors": [
                {"feature": name, "shap_value": round(float(val), 4)}
                for name, val in contributions[:10]
            ],
        }
