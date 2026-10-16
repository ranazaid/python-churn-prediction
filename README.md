# Python — Data / AI: Customer Churn Prediction

An end-to-end machine learning pipeline built with pandas and scikit-learn. It
generates a realistic synthetic customer dataset, explores it, compares two
models with cross-validation, evaluates the better one and saves it.

## Run

```bash
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
python churn_model.py
python churn_model.py --rows 20000 --seed 7
```

The run produces:
- churn rates by segment (contract, payment method, internet service)
- cross-validated ROC-AUC for Logistic Regression and Random Forest
- a classification report and the top feature drivers for the best model
- `output/churn_model.joblib` and `output/test_predictions.csv`, with customers ranked by churn risk

## Concepts shown
- Exploratory analysis with pandas `groupby`/`agg`
- A `ColumnTransformer` that scales numeric columns and one-hot encodes categorical ones
- `Pipeline`, so preprocessing never leaks across CV folds
- Stratified K-fold model comparison and class-imbalance handling
- Model persistence with `joblib`

## Loading the saved model

```python
import joblib, pandas as pd
model = joblib.load("output/churn_model.joblib")
model.predict_proba(pd.DataFrame([{
    "tenure_months": 3, "monthly_charges": 95, "support_tickets": 4, "data_usage_gb": 20,
    "contract": "month-to-month", "payment_method": "e-check", "internet_service": "fiber",
}]))[:, 1]
```

---

Portfolio reference implementation by [Zaid Mubbashar](https://github.com/ranazaid) · [zaid-mubbashar.vercel.app](https://zaid-mubbashar.vercel.app)
