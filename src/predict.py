import json
import joblib
import numpy as np
import pandas as pd
from typing import List

# Load trained model
model = joblib.load("results/model.pkl")  # adjust path if needed

# Load training feature columns from cleaned dataset to ensure alignment
try:
    training_columns_all: List[str] = list(pd.read_csv("data/cleaned_data.csv", nrows=0).columns)
except FileNotFoundError:
    # Fallback to results path if running from project root variations
    training_columns_all = list(pd.read_csv("./data/cleaned_data.csv", nrows=0).columns)

training_feature_columns: List[str] = [c for c in training_columns_all if c != "Churn"]

# Columns used during preprocessing in training
BINARY_YES_NO_COLUMNS = [
    "Partner",
    "Dependents",
    "PhoneService",
    "PaperlessBilling",
]

# LabelEncoder was also applied to gender (Female/Male) in preprocessing
BINARY_GENDER_COLUMN = "gender"

MULTI_CLASS_COLUMNS = [
    "MultipleLines",
    "InternetService",
    "OnlineSecurity",
    "OnlineBackup",
    "DeviceProtection",
    "TechSupport",
    "StreamingTV",
    "StreamingMovies",
    "Contract",
    "PaymentMethod",
]

NUMERIC_COLUMNS = [
    "SeniorCitizen",
    "tenure",
    "MonthlyCharges",
    "TotalCharges",
]


def _preprocess_input(raw_df: pd.DataFrame) -> pd.DataFrame:
    """Mirror the training-time preprocessing so the model receives aligned features.

    Steps:
    - Drop customerID if present
    - Coerce TotalCharges to numeric
    - Encode binary columns to 0/1 consistently with training
    - One-hot encode multi-class columns (drop_first=True as in training)
    - Ensure numeric dtypes for numeric columns
    - Align to training_feature_columns, adding missing columns with 0 and ordering columns
    """
    df = raw_df.copy()

    # Drop non-feature id if present
    if "customerID" in df.columns:
        df = df.drop(columns=["customerID"])  # dropped in training

    # Coerce TotalCharges to numeric (training replaced blanks with NaN and coerced)
    if "TotalCharges" in df.columns:
        df["TotalCharges"] = pd.to_numeric(df["TotalCharges"], errors="coerce")

    # Encode binary yes/no columns
    for col in BINARY_YES_NO_COLUMNS:
        if col in df.columns:
            df[col] = (
                df[col]
                .replace({"No": 0, "Yes": 1, "no": 0, "yes": 1, False: 0, True: 1})
                .astype("Int64")
            )

    # Encode gender consistent with LabelEncoder alphabetical ordering: Female->0, Male->1
    if BINARY_GENDER_COLUMN in df.columns:
        df[BINARY_GENDER_COLUMN] = (
            df[BINARY_GENDER_COLUMN]
            .replace({"Female": 0, "Male": 1, "female": 0, "male": 1})
            .astype("Int64")
        )

    # One-hot encode multi-class categoricals; drop_first=True as in training
    present_multi_cols = [c for c in MULTI_CLASS_COLUMNS if c in df.columns]
    if present_multi_cols:
        df = pd.get_dummies(df, columns=present_multi_cols, drop_first=True)

    # Ensure numeric dtypes for known numeric columns
    for col in NUMERIC_COLUMNS:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")

    # Align to training feature columns: add any missing as 0, drop extras, correct order
    aligned = pd.DataFrame(0, index=df.index, columns=training_feature_columns, dtype=float)
    for col in df.columns.intersection(training_feature_columns):
        aligned[col] = df[col]

    # Fill NaNs introduced by coercion
    aligned = aligned.fillna(0)

    return aligned[training_feature_columns]


def predict_churn(new_data: pd.DataFrame):
    processed = _preprocess_input(new_data)
    # If model supports predict_proba, return both label and probability of churn
    if hasattr(model, "predict_proba"):
        proba = model.predict_proba(processed)[:, 1]
        labels = (proba >= 0.5).astype(int)
        return labels, proba
    else:
        labels = model.predict(processed)
        return labels, None


if __name__ == "__main__":
    # Allow optional JSON input via CLI for flexibility
    import argparse

    parser = argparse.ArgumentParser(description="Predict churn for new customers")
    parser.add_argument(
        "--json",
        type=str,
        help="JSON string for a single customer record",
    )
    args = parser.parse_args()

    if args.json:
        record = json.loads(args.json)
        new_customer = pd.DataFrame([record])
    else:
        # Example new customer data (raw, unencoded)
        new_customer = pd.DataFrame([
            {
                "gender": "Female",
                "SeniorCitizen": 0,
                "Partner": "Yes",
                "Dependents": "No",
                "tenure": 12,
                "PhoneService": "Yes",
                "MultipleLines": "No phone service",
                "InternetService": "Fiber optic",
                "OnlineSecurity": "No",
                "OnlineBackup": "No",
                "DeviceProtection": "No",
                "TechSupport": "No",
                "StreamingTV": "No",
                "StreamingMovies": "No",
                "Contract": "Month-to-month",
                "PaperlessBilling": "Yes",
                "PaymentMethod": "Electronic check",
                "MonthlyCharges": 70.35,
                "TotalCharges": 850.5,
            }
        ])

    labels, proba = predict_churn(new_customer)
    label_str = "Yes" if int(labels[0]) == 1 else "No"
    if proba is not None:
        print(f"Churn Prediction: {label_str} (prob={proba[0]:.3f})")
    else:
        print(f"Churn Prediction: {label_str}")
