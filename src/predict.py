import pandas as pd
import joblib

# Load trained model
model = joblib.load("../results/model.pkl")

# Feature columns from training (one-hot)
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

# Collect input from user
print("Enter customer details (numeric for simplicity, or 0/1 for yes/no):")
gender = int(input("Gender (Male=1, Female=0): "))
SeniorCitizen = int(input("Senior Citizen (Yes=1, No=0): "))
Partner = int(input("Partner (Yes=1, No=0): "))
Dependents = int(input("Dependents (Yes=1, No=0): "))
tenure = int(input("Tenure (months): "))
PhoneService = int(input("Phone Service (Yes=1, No=0): "))
PaperlessBilling = int(input("Paperless Billing (Yes=1, No=0): "))
MonthlyCharges = float(input("Monthly Charges: "))
TotalCharges = float(input("Total Charges: "))

# Start with all zeros
new_df = pd.DataFrame([{col: 0 for col in columns}])

# Fill numeric features
new_df['gender'] = gender
new_df['SeniorCitizen'] = SeniorCitizen
new_df['Partner'] = Partner
new_df['Dependents'] = Dependents
new_df['tenure'] = tenure
new_df['PhoneService'] = PhoneService
new_df['PaperlessBilling'] = PaperlessBilling
new_df['MonthlyCharges'] = MonthlyCharges
new_df['TotalCharges'] = TotalCharges

# Ask categorical options and convert to one-hot
multiple_lines = input("Multiple Lines (No/Yes/No phone service): ")
if multiple_lines == "Yes":
    new_df['MultipleLines_Yes'] = 1
elif multiple_lines == "No phone service":
    new_df['MultipleLines_No phone service'] = 1

internet_service = input("Internet Service (Fiber optic/No): ")
if internet_service == "Fiber optic":
    new_df['InternetService_Fiber optic'] = 1
elif internet_service == "No":
    new_df['InternetService_No'] = 1

# Add more similar inputs for other categorical columns...
# OnlineSecurity, OnlineBackup, DeviceProtection, TechSupport, StreamingTV, StreamingMovies, Contract, PaymentMethod

# Predict
prediction = model.predict(new_df)
print("Prediction: Churn" if prediction[0] == 1 else "Prediction: No Churn")
