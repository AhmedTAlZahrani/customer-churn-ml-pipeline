from contextlib import asynccontextmanager
from pathlib import Path

import joblib
import pandas as pd
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel

MODEL_PATH = Path("models/best_model.pkl")
FE_PATH = Path("models/feature_engineer.pkl")

model = None
feature_engineer = None


@asynccontextmanager
async def lifespan(app: FastAPI):
    global model, feature_engineer
    if MODEL_PATH.exists():
        model = joblib.load(MODEL_PATH)
    if FE_PATH.exists():
        feature_engineer = joblib.load(FE_PATH)
    yield


app = FastAPI(
    title="Customer Churn Prediction API",
    description="Predict customer churn probability with SHAP explanations.",
    version="1.0.0",
    lifespan=lifespan,
)


class CustomerFeatures(BaseModel):
    tenure: int = 12
    MonthlyCharges: float = 70.0
    TotalCharges: float = 840.0
    Contract: str = "Month-to-month"
    InternetService: str = "Fiber optic"
    PaymentMethod: str = "Electronic check"
    gender: str = "Male"
    SeniorCitizen: int = 0
    Partner: str = "No"
    Dependents: str = "No"
    PhoneService: str = "Yes"
    OnlineSecurity: str = "No"
    TechSupport: str = "No"
    StreamingTV: str = "No"


@app.get("/model-info")
def model_info():
    if model is None:
        raise HTTPException(status_code=503, detail="Model not loaded")
    return {
        "model_type": type(model).__name__,
        "model_path": str(MODEL_PATH),
    }


# TODO: add caching for repeated predictions
@app.post("/predict")
def predict(features: CustomerFeatures):
    """Predict churn probability for a customer."""
    if model is None:
        raise HTTPException(status_code=503, detail="Model not loaded")

    df = pd.DataFrame([features.model_dump()])

    if feature_engineer is not None:
        df = feature_engineer.transform(df)

    proba = model.predict_proba(df)[0]
    prediction = int(proba[1] >= 0.5)

    return {
        "churn_prediction": prediction,
        "churn_probability": round(float(proba[1]), 4),
        "retain_probability": round(float(proba[0]), 4),
        "risk_level": "HIGH" if proba[1] > 0.7 else ("MEDIUM" if proba[1] > 0.4 else "LOW"),
    }
