import pandas as pd
import joblib

# Load trained model and feature columns
model = joblib.load("../results/model.pkl")
columns = [
    'gender', 'SeniorCitizen', 'Partner', 'Dependents', 'tenure', 'PhoneService',
    'PaperlessBilling', 'MonthlyCharges', 'TotalCharges',
    'MultipleLines_No phone service', 'MultipleLines_Yes',
    'InternetService_Fiber optic', 'InternetService_No',
    'OnlineSecurity_No internet service', 'OnlineSecurity_Yes',
    'OnlineBackup_No internet service', 'OnlineBackup_Yes',
    'DeviceProtection_No internet service', 'DeviceProtection_Yes',
    'TechSupport_No internet service', 'TechSupport_Yes',
    'StreamingTV_No internet service', 'StreamingTV_Yes',
    'StreamingMovies_No internet service', 'StreamingMovies_Yes',
    'Contract_One year', 'Contract_Two year',
    'PaymentMethod_Credit card (automatic)', 'PaymentMethod_Electronic check', 'PaymentMethod_Mailed check'
]

# Mapping for categorical one-hot columns
CATEGORICAL_MAPS = {
    'MultipleLines': {
        "Yes": "MultipleLines_Yes",
        "No": "MultipleLines_No phone service",
        "No phone service": "MultipleLines_No phone service"
    },
    'InternetService': {
        "Fiber optic": "InternetService_Fiber optic",
        "No": "InternetService_No"
    },
    'OnlineSecurity': {
        "Yes": "OnlineSecurity_Yes",
        "No": "OnlineSecurity_No internet service",
        "No internet service": "OnlineSecurity_No internet service"
    },
    'OnlineBackup': {
        "Yes": "OnlineBackup_Yes",
        "No": "OnlineBackup_No internet service",
        "No internet service": "OnlineBackup_No internet service"
    },
    'DeviceProtection': {
        "Yes": "DeviceProtection_Yes",
        "No": "DeviceProtection_No internet service",
        "No internet service": "DeviceProtection_No internet service"
    },
    'TechSupport': {
        "Yes": "TechSupport_Yes",
        "No": "TechSupport_No internet service",
        "No internet service": "TechSupport_No internet service"
    },
    'StreamingTV': {
        "Yes": "StreamingTV_Yes",
        "No": "StreamingTV_No internet service",
        "No internet service": "StreamingTV_No internet service"
    },
    'StreamingMovies': {
        "Yes": "StreamingMovies_Yes",
        "No": "StreamingMovies_No internet service",
        "No internet service": "StreamingMovies_No internet service"
    },
    'Contract': {
        "Month-to-month": None,
        "One year": "Contract_One year",
        "Two year": "Contract_Two year"
    },
    'PaymentMethod': {
        "Credit card (automatic)": "PaymentMethod_Credit card (automatic)",
        "Electronic check": "PaymentMethod_Electronic check",
        "Mailed check": "PaymentMethod_Mailed check"
    }
}

def prepare_customer(row):
    """Convert a single customer dict to one-hot DataFrame aligned with training columns"""
    df = pd.DataFrame([{col: 0 for col in columns}])

    # Numeric features
    numeric_cols = ['gender', 'SeniorCitizen', 'Partner', 'Dependents',
                    'tenure', 'PhoneService', 'PaperlessBilling',
                    'MonthlyCharges', 'TotalCharges']
    for col in numeric_cols:
        df[col] = row[col]

    # Categorical features
    for cat_col, mapping in CATEGORICAL_MAPS.items():
        val = row.get(cat_col)
        if val is None:
            continue
        mapped_col = mapping.get(val)
        if mapped_col:
            df[mapped_col] = 1
    return df

def predict_interactive():
    """Interactive input mode for a single customer"""
    row = {}
    print("Enter customer details (numeric for simplicity, or 0/1 for yes/no):")
    row['gender'] = int(input("Gender (Male=1, Female=0): "))
    row['SeniorCitizen'] = int(input("Senior Citizen (Yes=1, No=0): "))
    row['Partner'] = int(input("Partner (Yes=1, No=0): "))
    row['Dependents'] = int(input("Dependents (Yes=1, No=0): "))
    row['tenure'] = int(input("Tenure (months): "))
    row['PhoneService'] = int(input("Phone Service (Yes=1, No=0): "))
    row['PaperlessBilling'] = int(input("Paperless Billing (Yes=1, No=0): "))
    row['MonthlyCharges'] = float(input("Monthly Charges: "))
    row['TotalCharges'] = float(input("Total Charges: "))

    # Categorical
    for cat in ['MultipleLines','InternetService','OnlineSecurity','OnlineBackup','DeviceProtection',
                'TechSupport','StreamingTV','StreamingMovies','Contract','PaymentMethod']:
        val = input(f"{cat} ({'/'.join(CATEGORICAL_MAPS[cat].keys())}): ")
        row[cat] = val

    df_prepared = prepare_customer(row)
    pred = model.predict(df_prepared)[0]
    print("Prediction: Churn" if pred==1 else "Prediction: No Churn")

def predict_csv(file_path):
    """Batch prediction from CSV"""
    df_input = pd.read_csv(file_path)
    df_final = pd.DataFrame()
    for _, row in df_input.iterrows():
        df_prepared = prepare_customer(row)
        df_final = pd.concat([df_final, df_prepared], ignore_index=True)
    df_input['Prediction'] = model.predict(df_final)
    df_input['Prediction'] = df_input['Prediction'].map({1: 'Churn', 0: 'No Churn'})
    print(df_input)
    df_input.to_csv("predictions.csv", index=False)
    print("Predictions saved to predictions.csv")

if __name__ == "__main__":
    mode = input("Select mode (interactive/csv): ").strip().lower()
    if mode == "interactive":
        predict_interactive()
    elif mode == "csv":
        path = input("Enter CSV file path: ")
        predict_csv(path)
    else:
        print("Invalid mode. Choose 'interactive' or 'csv'.")
