# 🚀 Customer Churn Prediction

A powerful machine learning project built with Python and XGBoost, designed to predict whether a customer is likely to churn. This project allows users to input customer data and quickly get predictions, making it a great showcase for AI/ML internship portfolios.

---

## ✨ Key Features

🧮 **Accurate Churn Prediction:** Uses a trained XGBoost model to classify customers as churn or no churn.
📊 **Handles Categorical and Numeric Data:** Supports both one-hot encoded categorical features and numeric values.
💻 **Interactive Input:** Recruiters or users can enter customer details directly in the terminal for predictions.
📁 **CSV Input Support:** Predict churn for multiple customers at once using CSV files.
🧾 **Preprocessing & Feature Management:** Ensures new data matches the model’s training features, avoiding errors.
✅ **Ready-to-Use:** Includes trained model and all necessary feature columns for easy testing.

---

## 💻 Tech Stack

* **Backend / ML:** Python, Pandas, XGBoost, Joblib
* **Data Handling:** CSV, DataFrames
* **Environment Management:** Virtual environment (venv)

---

## ⚙️ Getting Started

Follow these steps to set up and run the Customer Churn Prediction project locally.

### Prerequisites

* Python 3.x installed
* Recommended: Virtual environment for dependency management

---

### 🔧 Installation

Clone the repository:

```bash
git clone https://github.com/AaryanR1508/customer-churn-prediction.git
cd customer-churn-prediction
```

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

### 🚀 Running the Application

#### **Option 1: Interactive Input**

Run the prediction script and input customer details:

```bash
python src/predict.py
```

Follow the prompts to enter numeric and categorical values. The program will output:

```
Prediction: Churn
```

or

```
Prediction: No Churn
```

#### **Option 2: CSV Input**

Prepare a CSV file with customer data in the `data/` folder and run:

```bash
python src/predict.py --csv data/sample_customers.csv
```

The script will output predictions for all rows in the CSV.

---

### 🔐 Environment Variables

You can configure paths to model or CSV files by editing `config.py` or `.env` (optional for advanced use).

---

### 🛤️ Roadmap (Optional Ideas)

* Web app interface with Flask/FastAPI
* Visual dashboard showing churn probabilities
* Automated retraining on new customer data
* Feature importance visualization

---

### 🧑‍💻 Author

Made with 💻 and ☕ by Anoop G.

---

