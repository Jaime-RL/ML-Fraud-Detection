# ML Fraud Detection & Classic ML Models

Machine learning projects in Python, centred on **credit card fraud detection** with a highly imbalanced dataset, plus several classic models implemented **from scratch** (gradient descent, logistic regression) and with **scikit-learn**.

## Highlights

- **SMOTE implemented from scratch** (k-NN + linear interpolation) and validated against `imbalanced-learn`.
- Leakage-free pipelines: stratified split first, then scaling and oversampling **only on the training set**.
- Metrics suited to imbalanced data: precision, recall, F1 and PR-AUC instead of accuracy alone.
- Gradient descent written by hand for linear and logistic regression.

## Projects

| Folder | Script | Description |
|---|---|---|
| `fraud_detection/` | `fraud_detector.py` | Fraud detection (0.17 % positive class): custom SMOTE, Logistic Regression and Random Forest |
| `classification/` | `customer_churn.py` | Customer churn prediction (IBM Telco) with SMOTE, Logistic Regression vs Random Forest |
| `classification/` | `logistic_regression_scratch.py` | Logistic regression with hand-written gradient descent and live decision curve |
| `regression/` | `house_price_prediction.py` | California housing prices: feature engineering, Linear Regression vs Random Forest (R²) |
| `regression/` | `linear_regression_scratch.py` | Simple linear regression (hours studied → score) with gradient descent |
| `regression/` | `gradient_descent_linear_regression.py` | Cost / gradient functions and batch gradient descent |

## Datasets

Datasets are not included in the repository. Download them and place the CSV next to the script:

| Script | Dataset | File |
|---|---|---|
| `fraud_detector.py` | [Credit Card Fraud Detection (Kaggle)](https://www.kaggle.com/datasets/mlg-ulb/creditcardfraud) | `creditcard.csv` |
| `customer_churn.py` | [Telco Customer Churn (Kaggle)](https://www.kaggle.com/datasets/blastchar/telco-customer-churn) | `Telco-Customer-Churn.csv` |
| `house_price_prediction.py` | [California Housing Prices (Kaggle)](https://www.kaggle.com/datasets/camnugent/california-housing-prices) | `housing.csv` |
| `linear_/logistic_regression_scratch.py` | [Student Study Hours (Kaggle)](https://www.kaggle.com/datasets/himanshunakrani/student-study-hours) | `score.csv` |

## Getting started

```bash
python -m venv .venv && source .venv/bin/activate   # Windows: .venv\Scripts\activate
pip install -r requirements.txt

python fraud_detection/fraud_detector.py --data path/to/creditcard.csv
```

Tested with Python 3.12, scikit-learn 1.x and imbalanced-learn 0.12+.

## Tech

Python · NumPy · pandas · scikit-learn · imbalanced-learn · Matplotlib
