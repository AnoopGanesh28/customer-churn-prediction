from pathlib import Path
import joblib
import pandas as pd

BASE_DIR = Path(__file__).resolve().parents[1]
MODEL_PATH = BASE_DIR / "results" / "model.pkl"
FEATURE_COLUMNS_PATH = BASE_DIR / "results" / "feature_columns.pkl"

model = joblib.load(MODEL_PATH)
columns = joblib.load(FEATURE_COLUMNS_PATH)

# Create a minimal valid input matching training columns
sample = {col: 0 for col in columns}

# Set some plausible numeric values
for key, value in {
    'gender': 1,
    'SeniorCitizen': 0,
    'Partner': 1,
    'Dependents': 0,
    'tenure': 12,
    'PhoneService': 1,
    'PaperlessBilling': 1,
    'MonthlyCharges': 70.5,
    'TotalCharges': 850.0,
}.items():
    if key in sample:
        sample[key] = value

# Turn on a couple of one-hot columns that should exist in columns
for oh in [
    'MultipleLines_Yes',
    'InternetService_Fiber optic',
    'OnlineBackup_Yes',
    'TechSupport_Yes',
    'StreamingTV_Yes',
]:
    if oh in sample:
        sample[oh] = 1

df = pd.DataFrame([sample])
pred = model.predict(df)[0]
print({"prediction": int(pred)})


