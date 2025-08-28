# 🚀 Customer Churn Prediction

A machine learning project built with **Python** and **XGBoost**, designed to predict whether a customer is likely to churn.  
This project allows users to input customer data and quickly get predictions, making it a great showcase for **AI/ML internship portfolios**.

---

## ✨ Key Features

- **Accurate Churn Prediction:** Uses a trained XGBoost model to classify customers as churn or no churn.  
- **Handles Categorical and Numeric Data:** Supports both one-hot encoded categorical features and numeric values.  
- **Interactive Input:** Enter customer details directly in the terminal for predictions.  
- **CSV Input Support:** Predict churn for multiple customers at once using CSV files.  
- **Preprocessing & Feature Management:** Ensures new data matches the model’s training features, avoiding errors.  
- **Ready-to-Use:** Includes trained model and all necessary feature columns for easy testing.  

---

## 🛠️ Tech Stack

- **Backend / ML:** Python, Pandas, XGBoost, Joblib  
- **Data Handling:** CSV, DataFrames  
- **Environment:** Virtual environment (venv)  

---

## 📂 Project Structure

```

CUSTOMER-CHURN-PREDICTION
│── data/
│   ├── cleaned\_data.csv
│   ├── Telco-Customer-Churn.csv
│   └── sample\_customers.csv
│
│── notebooks/
│   ├── data\_exploration.ipynb
│   ├── modeling.ipynb
│   ├── preprocessing.ipynb
│
│── results/
│   ├── model.pkl
│   ├── feature\_columns.pkl
│
│── src/
│   ├── **init**.py
│   ├── predict.py
│   ├── quick\_test.py
│   └── predictions.csv
│
│── venv/ (virtual environment)
│── .gitignore
│── README.md
│── requirements.txt

````

---

## ⚡ Getting Started

Follow these steps to set up and run the Customer Churn Prediction project locally.

### ✅ Prerequisites

- Python 3.x installed  
- Recommended: Virtual environment for dependency management  

---

### 🔧 Installation

Clone the repository:

```bash
git clone https://github.com/AnoopGanesh28/customer-churn-prediction.git
cd customer-churn-prediction
````

Create and activate a virtual environment:

```bash
python -m venv venv

# On Windows
.\venv\Scripts\activate

# On macOS/Linux
source venv/bin/activate
```

Install project dependencies:

```bash
pip install -r requirements.txt
```

---

## ▶️ Running the Application

There are two modes: **interactive (single row)** and **batch CSV (multiple rows)**.

### Option 1: Interactive Input (Single Prediction)

```bash
python src/predict.py --interactive
```

Follow the prompts to enter customer details.
The output will be:

```
Prediction: Churn
```

or

```
Prediction: No Churn
```

---

### Option 2: CSV Input (Batch Predictions)

Prepare a CSV file (e.g., `data/sample_customers.csv`) and run:

```bash
python src/predict.py --csv data/sample_customers.csv
```

* Input schema: see `data/sample_customers.csv`.
* Output file: predictions are saved as `predictions.csv` and also printed to the console.

---

## 📊 Testing & Validation

### Sample Data

A ready-to-run sample is included:

```
data/sample_customers.csv
```

Run:

```bash
python src/predict.py --csv data/sample_customers.csv
```

This will produce:

```
data/predictions.csv
```

with a `Prediction` column.

---

### Quick Sanity Test

To verify that the model and features load correctly:

```bash
python src/quick_test.py
```

This prints a small JSON-like line containing the predicted class.

---

## 🌱 Roadmap (Future Enhancements)

* Web app interface with Flask/FastAPI
* Visual dashboard showing churn probabilities
* Automated retraining on new customer data
* Feature importance visualization

---

## 👨‍💻 Author

Made with Python and perseverance by **Anoop Ganesh**



