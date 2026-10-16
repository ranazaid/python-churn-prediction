"""Customer churn prediction — end-to-end data/ML pipeline.

1. Generates a realistic synthetic telecom-style customer dataset (no download).
2. Explores it with pandas (summary stats and churn rate by segment).
3. Trains two scikit-learn pipelines (Logistic Regression vs Random Forest) using
   ColumnTransformer preprocessing and stratified cross-validation.
4. Evaluates the better model on a held-out test set (ROC-AUC, classification report).
5. Prints the top feature importances and saves the model plus a CSV of predictions.

Usage:
    python churn_model.py
    python churn_model.py --rows 20000 --seed 7
"""
from __future__ import annotations

import argparse
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, roc_auc_score
from sklearn.model_selection import StratifiedKFold, cross_val_score, train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

NUMERIC = ["tenure_months", "monthly_charges", "support_tickets", "data_usage_gb"]
CATEGORICAL = ["contract", "payment_method", "internet_service"]
TARGET = "churned"


def make_dataset(n: int, seed: int) -> pd.DataFrame:
    """Synthetic but plausible data: churn driven by contract, tenure, tickets and price."""
    rng = np.random.default_rng(seed)
    contract = rng.choice(["month-to-month", "one-year", "two-year"], n, p=[0.55, 0.25, 0.20])
    df = pd.DataFrame(
        {
            "tenure_months": rng.integers(1, 72, n),
            "monthly_charges": rng.normal(70, 25, n).clip(15, 150).round(2),
            "support_tickets": rng.poisson(1.2, n),
            "data_usage_gb": rng.gamma(2.0, 12.0, n).round(1),
            "contract": contract,
            "payment_method": rng.choice(["card", "bank", "e-check", "cash"], n),
            "internet_service": rng.choice(["fiber", "dsl", "none"], n, p=[0.45, 0.4, 0.15]),
        }
    )
    logit = (
        -1.2
        + 1.4 * (df.contract == "month-to-month")
        - 0.9 * (df.contract == "two-year")
        - 0.035 * df.tenure_months
        + 0.45 * df.support_tickets
        + 0.012 * (df.monthly_charges - 70)
        + 0.5 * (df.payment_method == "e-check")
        + 0.3 * (df.internet_service == "fiber")
    )
    df[TARGET] = (rng.random(n) < 1 / (1 + np.exp(-logit))).astype(int)
    return df


def explore(df: pd.DataFrame) -> None:
    print(f"Rows: {len(df):,}   Churn rate: {df[TARGET].mean():.1%}\n")
    print(df[NUMERIC].describe().round(1).to_string(), "\n")
    for col in CATEGORICAL:
        rates = df.groupby(col)[TARGET].agg(["mean", "count"]).sort_values("mean", ascending=False)
        rates["mean"] = (rates["mean"] * 100).round(1).astype(str) + "%"
        print(f"Churn by {col}:\n{rates.rename(columns={'mean': 'churn'}).to_string()}\n")


def build_pipelines(seed: int) -> dict[str, Pipeline]:
    pre = ColumnTransformer(
        [
            ("num", StandardScaler(), NUMERIC),
            ("cat", OneHotEncoder(handle_unknown="ignore"), CATEGORICAL),
        ]
    )
    return {
        "logistic_regression": Pipeline(
            [("pre", pre), ("clf", LogisticRegression(max_iter=1000, class_weight="balanced"))]
        ),
        "random_forest": Pipeline(
            [
                ("pre", pre),
                (
                    "clf",
                    RandomForestClassifier(
                        n_estimators=300, min_samples_leaf=5, class_weight="balanced",
                        random_state=seed, n_jobs=-1,
                    ),
                ),
            ]
        ),
    }


def feature_importances(model: Pipeline) -> pd.Series:
    names = model.named_steps["pre"].get_feature_names_out()
    clf = model.named_steps["clf"]
    values = clf.feature_importances_ if hasattr(clf, "feature_importances_") else np.abs(clf.coef_[0])
    return pd.Series(values, index=names).sort_values(ascending=False)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--rows", type=int, default=8000)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--out", type=Path, default=Path("output"))
    args = parser.parse_args()

    df = make_dataset(args.rows, args.seed)
    explore(df)

    X, y = df[NUMERIC + CATEGORICAL], df[TARGET]
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, stratify=y, random_state=args.seed
    )

    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=args.seed)
    scores = {}
    for name, pipe in build_pipelines(args.seed).items():
        auc = cross_val_score(pipe, X_train, y_train, cv=cv, scoring="roc_auc")
        scores[name] = (auc.mean(), pipe)
        print(f"CV ROC-AUC  {name:<20} {auc.mean():.3f} ± {auc.std():.3f}")

    best_name, (_, best) = max(scores.items(), key=lambda kv: kv[1][0])
    best.fit(X_train, y_train)
    proba = best.predict_proba(X_test)[:, 1]
    print(f"\nBest model: {best_name}")
    print(f"Test ROC-AUC: {roc_auc_score(y_test, proba):.3f}\n")
    print(classification_report(y_test, (proba >= 0.5).astype(int), target_names=["stay", "churn"]))

    print("Top features:")
    for feat, val in feature_importances(best).head(8).items():
        print(f"  {val:6.3f}  {feat}")

    args.out.mkdir(parents=True, exist_ok=True)
    joblib.dump(best, args.out / "churn_model.joblib")
    X_test.assign(actual=y_test, churn_probability=proba.round(3)).sort_values(
        "churn_probability", ascending=False
    ).to_csv(args.out / "test_predictions.csv", index=False)
    print(f"\nSaved model + predictions → {args.out}/")


if __name__ == "__main__":
    main()
