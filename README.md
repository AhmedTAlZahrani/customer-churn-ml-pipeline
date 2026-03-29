# Customer Churn ML Pipeline

![CI](https://github.com/AhmedTAlZahrani/customer-churn-ml-pipeline/actions/workflows/ci.yml/badge.svg)

![Python](https://img.shields.io/badge/Python-3.11-3776AB?style=for-the-badge&logo=python&logoColor=white)
![scikit-learn](https://img.shields.io/badge/scikit--learn-F7931E?style=for-the-badge&logo=scikit-learn&logoColor=white)
![XGBoost](https://img.shields.io/badge/XGBoost-189FDD?style=for-the-badge)
![SHAP](https://img.shields.io/badge/SHAP-Explainability-blueviolet?style=for-the-badge)
![Docker](https://img.shields.io/badge/Docker-2496ED?style=for-the-badge&logo=docker&logoColor=white)
![License](https://img.shields.io/badge/License-MIT-green?style=for-the-badge)

End-to-end machine learning pipeline for predicting customer churn in telecom. Covers the full ML lifecycle: data ingestion, feature engineering, model comparison, SHAP-based explainability, and deployment via FastAPI with Docker.

---

## Model Performance

| Model | Accuracy | Precision | Recall | F1-Score | AUC-ROC |
|-------|----------|-----------|--------|----------|---------|
| Logistic Regression | 80.3% | 66.2% | 54.8% | 59.9% | 0.838 |
| Random Forest | 79.5% | 64.7% | 48.2% | 55.2% | 0.826 |
| **XGBoost** | **82.1%** | **69.5%** | **57.3%** | **62.8%** | **0.871** |
| LightGBM | 81.6% | 68.1% | 56.9% | 62.0% | 0.864 |

> Best model: **XGBoost** with AUC-ROC = 0.871 on stratified 5-fold cross-validation.

### Top SHAP Features

1. `tenure` -- Long-tenure customers are far less likely to churn
2. `MonthlyCharges` -- Higher charges increase churn risk
3. `Contract_Month-to-month` -- Month-to-month contracts are high-risk
4. `InternetService_Fiber optic` -- Fiber optic users churn more often
5. `TotalCharges` -- Lower lifetime value correlates with churn

---

## Quick Start

### Install

```bash
git clone https://github.com/AhmedTAlZahrani/customer-churn-ml-pipeline.git
cd customer-churn-ml-pipeline
pip install -r requirements.txt
```

### Train Models

```python
from src.ingest import load_telco_data
from src.feature_engineering import FeatureEngineer
from src.model_training import ModelTrainer

X, y = load_telco_data("data/telco_churn.csv")
fe = FeatureEngineer()
X_processed = fe.fit_transform(X)

trainer = ModelTrainer()
comparison = trainer.benchmark_all(X_processed, y)
print(comparison)
```

### Serve with API

```bash
uvicorn api.main:app --host 0.0.0.0 --port 8000
```

```bash
curl -X POST http://localhost:8000/predict \
  -H "Content-Type: application/json" \
  -d '{"tenure": 12, "MonthlyCharges": 70.5, "Contract": "Month-to-month", "InternetService": "Fiber optic"}'
```

### Docker

```bash
docker build -t churn-predictor .
docker run -p 8000:8000 churn-predictor
```

---

## Dataset

[Telco Customer Churn](https://www.kaggle.com/datasets/blastchar/telco-customer-churn) from Kaggle (7,043 customers, 21 features). Place as `data/telco_churn.csv`.

---

## Project Structure

```
customer-churn-ml-pipeline/
├── src/
│   ├── __init__.py
│   ├── ingest.py              # Data ingestion and validation
│   ├── feature_engineering.py  # Feature transforms and encoding
│   ├── model_training.py       # Multi-model training and comparison
│   ├── metrics.py              # Metrics, confusion matrix, ROC curves
│   └── explainability.py       # SHAP global and local explanations
├── api/
│   ├── __init__.py
│   └── main.py                 # FastAPI prediction server
├── models/                     # Saved models (not tracked)
├── Dockerfile
├── requirements.txt
├── LICENSE
└── README.md
```

---

## Pipeline Architecture

```
Raw CSV
  |
  v
Data Loader --> Feature Engineering --> Train/Test Split
                                            |
                  +-------------------------+-------------------------+
                  |              |              |              |
              LogReg        RandomForest     XGBoost       LightGBM
                  |              |              |              |
                  +-------------------------+-------------------------+
                                            |
                                    Model Comparison
                                            |
                                     Best Model --> SHAP Explainability
                                            |
                                     Save (joblib)
                                            |
                                   FastAPI + Docker
```

---

## Running Tests

```bash
make test
```

Or run directly with coverage:

```bash
pytest tests/ --cov=src
```

---

## License

MIT License -- see [LICENSE](LICENSE) for details.
