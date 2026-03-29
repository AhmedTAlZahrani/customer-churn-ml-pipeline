import json
from pathlib import Path
from typing import Any, Dict, List, Optional, Union

import numpy as np
import pandas as pd
import plotly.express as px
from sklearn.metrics import (
    classification_report,
    confusion_matrix,
    roc_curve,
    auc,
    precision_recall_curve,
    average_precision_score,
)


class ModelEvaluator:
    """Evaluate classification model performance with visual reports."""

    @staticmethod
    def get_classification_report(y_true: Union[pd.Series, np.ndarray], y_pred: Union[pd.Series, np.ndarray]) -> pd.DataFrame:
        """Return classification report as a DataFrame."""
        report = classification_report(y_true, y_pred, output_dict=True)
        return pd.DataFrame(report).transpose().round(3)

    @staticmethod
    def plot_confusion_matrix(y_true: Union[pd.Series, np.ndarray], y_pred: Union[pd.Series, np.ndarray], labels: Optional[List[str]] = None) -> Any:
        """Generate an annotated confusion matrix heatmap."""
        labels = labels or ["No Churn", "Churn"]
        cm = confusion_matrix(y_true, y_pred)

        fig = px.imshow(
            cm, text_auto=True,
            x=labels, y=labels,
            color_continuous_scale="Blues",
            labels=dict(x="Predicted", y="Actual", color="Count"),
            title="Confusion Matrix",
        )
        fig.update_layout(template="plotly_dark", height=400)
        return fig

    @staticmethod
    def plot_roc_curve(y_true: Union[pd.Series, np.ndarray], y_proba: np.ndarray) -> Any:
        """Generate ROC curve with AUC score."""
        fpr, tpr, _ = roc_curve(y_true, y_proba)
        roc_auc = auc(fpr, tpr)

        fig = px.area(
            x=fpr, y=tpr,
            labels=dict(x="False Positive Rate", y="True Positive Rate"),
            title=f"ROC Curve (AUC = {roc_auc:.3f})",
        )
        fig.add_shape(type="line", x0=0, y0=0, x1=1, y1=1,
                       line=dict(dash="dash", color="gray"))
        fig.update_layout(template="plotly_dark", height=450)
        return fig

    @staticmethod
    def plot_precision_recall_curve(y_true: Union[pd.Series, np.ndarray], y_proba: np.ndarray) -> Any:
        """Generate Precision-Recall curve with average precision."""
        precision, recall, _ = precision_recall_curve(y_true, y_proba)
        ap = average_precision_score(y_true, y_proba)

        fig = px.area(
            x=recall, y=precision,
            labels=dict(x="Recall", y="Precision"),
            title=f"Precision-Recall Curve (AP = {ap:.3f})",
        )
        fig.update_layout(template="plotly_dark", height=450)
        return fig

    @staticmethod
    def save_metrics(metrics: Dict[str, Any], path: str = "output/metrics.json") -> None:
        """Save evaluation metrics to a JSON file."""
        output = Path(path)
        output.parent.mkdir(parents=True, exist_ok=True)
        with open(output, "w") as f:
            json.dump(metrics, f, indent=2, default=str)
        print(f"Metrics saved to {output}")
