import joblib
import pandas as pd

# Load trained model
model = joblib.load("results/model.pkl")  # adjust path if needed

def predict_churn(new_data: pd.DataFrame):
    return model.predict(new_data)

if __name__ == "__main__":
    # Example new customer data (make sure columns match training data)
    new_customer = pd.DataFrame([{
        "gender": "Female",
        "SeniorCitizen": 0,
        "Partner": "Yes",
        "Dependents": "No",
        "tenure": 12,
        "PhoneService": "Yes",
        "InternetService": "Fiber optic",
        "Contract": "Month-to-month",
        "PaperlessBilling": "Yes",
        "PaymentMethod": "Electronic check",
        "MonthlyCharges": 70.35,
        "TotalCharges": 850.5
    }])

    prediction = predict_churn(new_customer)
    print("Churn Prediction:", "Yes" if prediction[0] == 1 else "No")
