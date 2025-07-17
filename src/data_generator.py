"""Generate synthetic customer churn data for testing and demos.

Produces data matching the Telco Customer Churn dataset schema so it can
be consumed by the rest of the pipeline without a real CSV file.
"""

import numpy as np
import pandas as pd


def generate_churn_data(n_records=1000, random_state=42):
    """Generate synthetic churn data with realistic distributions.

    Args:
        n_records: Number of customer records to generate.
        random_state: Random seed for reproducibility.

    Returns:
        DataFrame with columns matching the Telco churn dataset.
    """
    rng = np.random.default_rng(random_state)

    tenure = rng.integers(0, 73, size=n_records)
    monthly_charges = rng.uniform(18.0, 120.0, size=n_records).round(2)
    total_charges = (tenure * monthly_charges + rng.normal(0, 50, n_records)).clip(0).round(2)

    contracts = rng.choice(
        ["Month-to-month", "One year", "Two year"],
        size=n_records,
        p=[0.5, 0.25, 0.25],
    )
    internet = rng.choice(
        ["DSL", "Fiber optic", "No"],
        size=n_records,
        p=[0.35, 0.45, 0.20],
    )
    payment = rng.choice(
        ["Electronic check", "Mailed check", "Bank transfer (automatic)", "Credit card (automatic)"],
        size=n_records,
        p=[0.35, 0.25, 0.20, 0.20],
    )
    yes_no = ["Yes", "No"]
    gender = rng.choice(["Male", "Female"], size=n_records)
    senior = rng.choice([0, 1], size=n_records, p=[0.84, 0.16])
    partner = rng.choice(yes_no, size=n_records)
    dependents = rng.choice(yes_no, size=n_records, p=[0.30, 0.70])
    phone_service = rng.choice(yes_no, size=n_records, p=[0.90, 0.10])
    online_security = rng.choice(["Yes", "No", "No internet service"], size=n_records, p=[0.30, 0.50, 0.20])
    tech_support = rng.choice(["Yes", "No", "No internet service"], size=n_records, p=[0.30, 0.50, 0.20])
    streaming_tv = rng.choice(["Yes", "No", "No internet service"], size=n_records, p=[0.35, 0.45, 0.20])

    # Churn probability influenced by contract type, tenure, and charges
    churn_prob = np.full(n_records, 0.26)
    churn_prob[contracts == "Month-to-month"] += 0.20
    churn_prob[contracts == "Two year"] -= 0.15
    churn_prob[tenure < 12] += 0.10
    churn_prob[tenure > 48] -= 0.10
    churn_prob[monthly_charges > 80] += 0.10
    churn_prob[internet == "Fiber optic"] += 0.08
    churn_prob[online_security == "Yes"] -= 0.05
    churn_prob = np.clip(churn_prob, 0.05, 0.95)
    churn = (rng.random(n_records) < churn_prob).astype(int)

    df = pd.DataFrame({
        "customerID": [f"CUST-{i:05d}" for i in range(n_records)],
        "gender": gender,
        "SeniorCitizen": senior,
        "Partner": partner,
        "Dependents": dependents,
        "tenure": tenure,
        "PhoneService": phone_service,
        "InternetService": internet,
        "OnlineSecurity": online_security,
        "TechSupport": tech_support,
        "StreamingTV": streaming_tv,
        "Contract": contracts,
        "PaymentMethod": payment,
        "MonthlyCharges": monthly_charges,
        "TotalCharges": total_charges,
        "Churn": churn,
    })

    return df
