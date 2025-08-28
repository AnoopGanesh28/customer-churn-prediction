# Customer Churn Prediction

A machine learning project built with Python and XGBoost to predict whether a customer is likely to churn. Users can input customer data and quickly get predictions.

---

## Key Features

- **Accurate churn prediction:** Uses a trained XGBoost model to classify customers as churn or no churn.
- **Handles categorical and numeric data:** Supports both one-hot encoded categorical features and numeric values.
- **Interactive input:** Enter customer details in the terminal for predictions.
- **CSV input support:** Predict churn for multiple customers at once using CSV files.
- **Preprocessing and feature management:** Ensures new data matches the model’s training features.
- **Ready to use:** Includes trained model and required feature columns for testing.

---

## Tech Stack

- **Backend / ML:** Python, Pandas, XGBoost, Joblib
- **Data Handling:** CSV, DataFrames
- **Environment:** Virtual environment (venv)

---

## Getting Started

Follow these steps to set up and run the Customer Churn Prediction project locally.

### Prerequisites

* Python 3.x installed
* Recommended: Virtual environment for dependency management

---

### Installation

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

### Running the Application

There are two modes: interactive (single row) and batch CSV (multiple rows).

#### Option 1: Interactive input (single prediction)

```bash
python src/predict.py --interactive
```

Follow the prompts to enter values; you will see either:
`Prediction: Churn` or `Prediction: No Churn`.

#### Option 2: CSV input (batch predictions)

```bash
python src/predict.py --csv data/sample_customers.csv
```

- Input schema: see the provided sample at `data/sample_customers.csv`.
- Output file: predictions are saved next to the input as `predictions.csv` and also printed to the console.

---

### Sample data

A ready-to-run sample is included at:

```
data/sample_customers.csv
```

Use it with:

```bash
python src/predict.py --csv data/sample_customers.csv
```

This will produce `data/predictions.csv` with a `Prediction` column.

### Quick sanity test

To verify the model and features load correctly and produce a prediction:

```bash
python src/quick_test.py
```

This prints a small JSON-like line containing the predicted class.

### Environment and paths

Artifacts are expected at:

- `results/model.pkl`
- `results/feature_columns.pkl`

These are loaded automatically by `src/predict.py`. Do not move them unless you update the script.

---

### Roadmap (optional)

* Web app interface with Flask/FastAPI
* Visual dashboard showing churn probabilities
* Automated retraining on new customer data
* Feature importance visualization

---

### Author

Made by Anoop G.

---

