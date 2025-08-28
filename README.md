# Customer Churn Prediction

<<<<<<< HEAD
A machine learning project built with Python and XGBoost to predict whether a customer is likely to churn. Users can input customer data and quickly get predictions.
=======
A machine learning project built with Python and XGBoost, designed to predict whether a customer is likely to churn. This project allows users to input customer data and quickly get predictions, making it a great showcase for AI/ML internship portfolios.
>>>>>>> 920fcd2594f02663696c3999623585405552556e

---

## Key Features

<<<<<<< HEAD
- **Accurate churn prediction:** Uses a trained XGBoost model to classify customers as churn or no churn.
- **Handles categorical and numeric data:** Supports both one-hot encoded categorical features and numeric values.
- **Interactive input:** Enter customer details in the terminal for predictions.
- **CSV input support:** Predict churn for multiple customers at once using CSV files.
- **Preprocessing and feature management:** Ensures new data matches the model’s training features.
- **Ready to use:** Includes trained model and required feature columns for testing.
=======
* **Accurate Churn Prediction:** Uses a trained XGBoost model to classify customers as churn or no churn.
* **Handles Categorical and Numeric Data:** Supports both one-hot encoded categorical features and numeric values.
* **Interactive Input:** Recruiters or users can enter customer details directly in the terminal for predictions.
* **CSV Input Support:** Predict churn for multiple customers at once using CSV files.
* **Preprocessing & Feature Management:** Ensures new data matches the model’s training features, avoiding errors.
* **Ready-to-Use:** Includes trained model and all necessary feature columns for easy testing.
>>>>>>> 920fcd2594f02663696c3999623585405552556e

---

## Tech Stack

- **Backend / ML:** Python, Pandas, XGBoost, Joblib
- **Data Handling:** CSV, DataFrames
- **Environment:** Virtual environment (venv)

---

<<<<<<< HEAD
=======
## Project Structure

```
CUSTOMER-CHURN-PREDICTION
│── data/
│   ├── cleaned_data.csv
│   ├── Telco-Customer-Churn.csv
│   └── .ipynb_checkpoints/
│
│── notebooks/
│   ├── data_exploration.ipynb
│   ├── modeling.ipynb
│   ├── preprocessing.ipynb
│   └── .ipynb_checkpoints/
│
│── results/
│
│── src/
│   ├── __init__.py
│   ├── evaluate_model.ipynb
│   ├── predict.py
│   ├── predictions.csv
│   └── .ipynb_checkpoints/
│
│── venv/ (virtual environment)
│
│── .gitignore
│── pyvenv.cfg
│── README.md
│── requirements.txt
```

---

>>>>>>> 920fcd2594f02663696c3999623585405552556e
## Getting Started

Follow these steps to set up and run the Customer Churn Prediction project locally.

### Prerequisites

* Python 3.x installed
* Recommended: Virtual environment for dependency management

---

### Installation

Clone the repository:

```bash
git clone https://github.com/AnoopGanesh28/customer-churn-prediction.git
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

<<<<<<< HEAD
### Running the Application

There are two modes: interactive (single row) and batch CSV (multiple rows).
=======
## Running the Application

### Option 1: Interactive Input
>>>>>>> 920fcd2594f02663696c3999623585405552556e

#### Option 1: Interactive input (single prediction)

```bash
python src/predict.py --interactive
```

<<<<<<< HEAD
Follow the prompts to enter values; you will see either:
`Prediction: Churn` or `Prediction: No Churn`.

#### Option 2: CSV input (batch predictions)
=======
Follow the prompts to enter numeric and categorical values. The program will output either:

```
Prediction: Churn
```

or

```
Prediction: No Churn
```

### Option 2: CSV Input

Prepare a CSV file with customer data in the `data/` folder and run:
>>>>>>> 920fcd2594f02663696c3999623585405552556e

```bash
python src/predict.py --csv data/sample_customers.csv
```

- Input schema: see the provided sample at `data/sample_customers.csv`.
- Output file: predictions are saved next to the input as `predictions.csv` and also printed to the console.

---

<<<<<<< HEAD
### Sample data
=======
## Environment Variables
>>>>>>> 920fcd2594f02663696c3999623585405552556e

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

<<<<<<< HEAD
### Roadmap (optional)
=======
## Roadmap (Future Enhancements)
>>>>>>> 920fcd2594f02663696c3999623585405552556e

* Web app interface with Flask/FastAPI
* Visual dashboard showing churn probabilities
* Automated retraining on new customer data
* Feature importance visualization

---

<<<<<<< HEAD
### Author

Made by Anoop G.
=======
## Author

Made with Python and perseverance by **Anoop Ganesh**.
>>>>>>> 920fcd2594f02663696c3999623585405552556e

---


