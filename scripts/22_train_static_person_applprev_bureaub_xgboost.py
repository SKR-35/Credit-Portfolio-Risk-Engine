from pathlib import Path
import json
import joblib

import numpy as np
import pandas as pd
import polars as pl
import xgboost as xgb

from sklearn.metrics import roc_auc_score
from sklearn.model_selection import train_test_split

DATA_PATH = Path(
    "data/processed/train_static_person_applprev_bureaub_features.parquet"
)

MODEL_DIR = Path("models")
REPORT_DIR = Path("reports")

MODEL_DIR.mkdir(exist_ok=True)
REPORT_DIR.mkdir(exist_ok=True)

print("Loading dataset...")
df = pl.read_parquet(DATA_PATH)

print(df.shape)

pdf = df.to_pandas()

target_col = "target"
drop_cols = ["case_id", "target"]

X = pdf.drop(columns=drop_cols)
y = pdf[target_col]

print("Encoding categorical and boolean features...")

encoded_cols = []

for col in X.columns:

    if X[col].dtype == "object" or str(X[col].dtype) == "category":

        X[col] = (
            X[col]
            .astype("string")
            .fillna("__MISSING__")
        )

        X[col] = (
            X[col]
            .astype("category")
            .cat.codes
            .astype("int32")
        )

        encoded_cols.append(col)

    elif str(X[col].dtype) in ["bool", "boolean"]:

        X[col] = (
            X[col]
            .fillna(False)
            .astype("int8")
        )

    else:

        X[col] = pd.to_numeric(
            X[col],
            errors="coerce"
        )

print(
    f"Encoded categorical features: {len(encoded_cols)}"
)

X = X.replace(
    [np.inf, -np.inf],
    np.nan
)

X_train, X_valid, y_train, y_valid = train_test_split(
    X,
    y,
    test_size=0.2,
    stratify=y,
    random_state=42
)

neg = (y_train == 0).sum()
pos = (y_train == 1).sum()

scale_pos_weight = neg / pos

print(
    f"scale_pos_weight: {scale_pos_weight:.4f}"
)

print(
    "Training XGBoost static + person + applprev + bureau_b model..."
)

model = xgb.XGBClassifier(
    objective="binary:logistic",
    eval_metric="auc",
    n_estimators=800,
    learning_rate=0.04,
    max_depth=6,
    min_child_weight=10,
    subsample=0.85,
    colsample_bytree=0.85,
    reg_alpha=0.001,
    reg_lambda=1.0,
    scale_pos_weight=scale_pos_weight,
    tree_method="hist",
    random_state=42,
    n_jobs=-1,
)

model.fit(
    X_train,
    y_train,
    eval_set=[(X_valid, y_valid)],
    verbose=50,
)

preds = model.predict_proba(
    X_valid
)[:, 1]

auc = roc_auc_score(
    y_valid,
    preds
)

gini = 2 * auc - 1

print("=" * 80)
print("STATIC + PERSON + APPLPREV + BUREAU_B XGBOOST SUMMARY")
print("=" * 80)

print(
    f"Validation AUC:  {auc:.6f}"
)

print(
    f"Validation Gini: {gini:.6f}"
)

importance = pd.DataFrame(
    {
        "feature": X.columns,
        "importance": model.feature_importances_,
    }
)

importance = (
    importance
    .sort_values(
        "importance",
        ascending=False
    )
)

print("\nTOP 30 FEATURES\n")
print(
    importance.head(30)
)

model_path = (
    MODEL_DIR /
    "static_person_applprev_bureaub_xgboost.joblib"
)

importance_path = (
    REPORT_DIR /
    "feature_importance_static_person_applprev_bureaub.csv"
)

summary_path = (
    REPORT_DIR /
    "static_person_applprev_bureaub_summary.json"
)

joblib.dump(
    model,
    model_path
)

importance.to_csv(
    importance_path,
    index=False
)

summary = {
    "dataset": str(DATA_PATH),
    "rows": int(df.height),
    "columns": int(df.width),
    "validation_auc": float(auc),
    "validation_gini": float(gini),
    "encoded_categorical_features": len(encoded_cols),
    "scale_pos_weight": float(scale_pos_weight),
}

with open(
    summary_path,
    "w",
    encoding="utf-8",
) as f:

    json.dump(
        summary,
        f,
        indent=2
    )

print("\nSaved:")
print(f"- {model_path}")
print(f"- {importance_path}")
print(f"- {summary_path}")