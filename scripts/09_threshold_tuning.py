"""
09_threshold_tuning.py

Tune the decision threshold for the selected credit risk model.

Current modelling decision:
- XGBoost was the strongest candidate in 08_compare_models.py.
- This script retrains the XGBoost pipeline on the same train/validation split,
  saves the full sklearn Pipeline as a joblib artifact, and evaluates multiple
  probability thresholds.

Why this script exists:
- AUC/Gini measure ranking power.
- Business decisions need a cutoff threshold.
- The default threshold 0.50 is often too high for imbalanced credit-risk data.

Outputs:
    models/final_xgboost_pipeline.joblib
    reports/threshold_metrics.csv
    reports/best_threshold.json
    reports/threshold_precision_recall.png
    reports/threshold_f1.png
    reports/threshold_confusion_counts.png

Usage:
    python scripts/09_threshold_tuning.py

Optional:
    python scripts/09_threshold_tuning.py --sample 300000
    python scripts/09_threshold_tuning.py --min-threshold 0.01 --max-threshold 0.50 --step 0.01
"""

from __future__ import annotations

import argparse
import json
import warnings
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import polars as pl

from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.metrics import (
    precision_score,
    recall_score,
    f1_score,
    confusion_matrix,
    roc_auc_score,
)
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

try:
    from xgboost import XGBClassifier
except Exception as exc:
    raise ImportError(
        "XGBoost is required for this script. Install it with: pip install xgboost"
    ) from exc


warnings.filterwarnings("ignore")

DATA_PATH = Path("data/processed/train_static_features.parquet")
REPORT_DIR = Path("reports")
MODEL_DIR = Path("models")

REPORT_DIR.mkdir(parents=True, exist_ok=True)
MODEL_DIR.mkdir(parents=True, exist_ok=True)


def build_preprocessor(X: pd.DataFrame) -> tuple[ColumnTransformer, list[str], list[str]]:
    categorical_cols = [
        col
        for col in X.columns
        if X[col].dtype == "object" or str(X[col].dtype) == "category"
    ]
    numeric_cols = [col for col in X.columns if col not in categorical_cols]

    numeric_pipe = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="median")),
            ("scaler", StandardScaler(with_mean=False)),
        ]
    )

    categorical_pipe = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="most_frequent")),
            (
                "onehot",
                OneHotEncoder(
                    handle_unknown="ignore",
                    min_frequency=50,
                    sparse_output=True,
                ),
            ),
        ]
    )

    preprocessor = ColumnTransformer(
        transformers=[
            ("num", numeric_pipe, numeric_cols),
            ("cat", categorical_pipe, categorical_cols),
        ],
        remainder="drop",
        sparse_threshold=0.3,
    )

    return preprocessor, numeric_cols, categorical_cols


def evaluate_thresholds(
    y_true: pd.Series,
    y_proba: np.ndarray,
    thresholds: np.ndarray,
) -> pd.DataFrame:
    rows = []
    n_total = len(y_true)
    n_defaults = int(y_true.sum())
    n_non_defaults = int(n_total - n_defaults)

    for threshold in thresholds:
        y_pred = (y_proba >= threshold).astype(int)

        tn, fp, fn, tp = confusion_matrix(y_true, y_pred).ravel()

        precision = precision_score(y_true, y_pred, zero_division=0)
        recall = recall_score(y_true, y_pred, zero_division=0)
        f1 = f1_score(y_true, y_pred, zero_division=0)

        predicted_default_rate = float((tp + fp) / n_total)
        predicted_approval_rate = float((tn + fn) / n_total)

        false_positive_rate = float(fp / n_non_defaults) if n_non_defaults else np.nan
        false_negative_rate = float(fn / n_defaults) if n_defaults else np.nan

        rows.append(
            {
                "threshold": float(threshold),
                "precision": precision,
                "recall": recall,
                "f1": f1,
                "tn": int(tn),
                "fp": int(fp),
                "fn": int(fn),
                "tp": int(tp),
                "predicted_default_rate": predicted_default_rate,
                "predicted_approval_rate": predicted_approval_rate,
                "false_positive_rate": false_positive_rate,
                "false_negative_rate": false_negative_rate,
                "default_capture_rate": recall,
            }
        )

    return pd.DataFrame(rows)


def save_threshold_plots(metrics: pd.DataFrame) -> None:
    import matplotlib.pyplot as plt

    plt.figure(figsize=(9, 6))
    plt.plot(metrics["threshold"], metrics["precision"], label="Precision")
    plt.plot(metrics["threshold"], metrics["recall"], label="Recall")
    plt.xlabel("Threshold")
    plt.ylabel("Score")
    plt.title("Precision and Recall by Threshold")
    plt.legend()
    plt.tight_layout()
    plt.savefig(REPORT_DIR / "threshold_precision_recall.png", dpi=150)
    plt.close()

    plt.figure(figsize=(9, 6))
    plt.plot(metrics["threshold"], metrics["f1"], label="F1")
    plt.xlabel("Threshold")
    plt.ylabel("F1")
    plt.title("F1 by Threshold")
    plt.legend()
    plt.tight_layout()
    plt.savefig(REPORT_DIR / "threshold_f1.png", dpi=150)
    plt.close()

    plt.figure(figsize=(9, 6))
    plt.plot(metrics["threshold"], metrics["tp"], label="True Positives")
    plt.plot(metrics["threshold"], metrics["fp"], label="False Positives")
    plt.plot(metrics["threshold"], metrics["fn"], label="False Negatives")
    plt.xlabel("Threshold")
    plt.ylabel("Count")
    plt.title("Confusion Matrix Counts by Threshold")
    plt.legend()
    plt.tight_layout()
    plt.savefig(REPORT_DIR / "threshold_confusion_counts.png", dpi=150)
    plt.close()


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--sample",
        type=int,
        default=None,
        help="Optional row sample for faster threshold tuning tests.",
    )
    parser.add_argument("--min-threshold", type=float, default=0.01)
    parser.add_argument("--max-threshold", type=float, default=0.50)
    parser.add_argument("--step", type=float, default=0.01)
    parser.add_argument("--random-state", type=int, default=42)
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    print("Loading dataset...")
    df = pl.read_parquet(DATA_PATH).to_pandas()
    print(f"Dataset shape: {df.shape}")

    if args.sample is not None and args.sample < len(df):
        print(f"Sampling {args.sample:,} rows for faster threshold tuning...")
        df = (
            df.groupby("target", group_keys=False)
            .apply(lambda x: x.sample(frac=args.sample / len(df), random_state=args.random_state))
            .sample(frac=1.0, random_state=args.random_state)
            .reset_index(drop=True)
        )
        print(f"Sampled shape: {df.shape}")

    target_col = "target"
    drop_cols = ["case_id", "target"]

    X = df.drop(columns=drop_cols)
    y = df[target_col]

    X_train, X_valid, y_train, y_valid = train_test_split(
        X,
        y,
        test_size=0.2,
        random_state=args.random_state,
        stratify=y,
    )

    preprocessor, numeric_cols, categorical_cols = build_preprocessor(X_train)

    print(f"Numeric features: {len(numeric_cols)}")
    print(f"Categorical features: {len(categorical_cols)}")

    print("\nTraining selected model: XGBoost...")
    xgb = XGBClassifier(
        n_estimators=600,
        learning_rate=0.03,
        max_depth=6,
        subsample=0.85,
        colsample_bytree=0.85,
        objective="binary:logistic",
        eval_metric="auc",
        tree_method="hist",
        random_state=args.random_state,
        n_jobs=-1,
    )

    pipeline = Pipeline(
        steps=[
            ("preprocessor", preprocessor),
            ("model", xgb),
        ]
    )

    pipeline.fit(X_train, y_train)

    print("Generating validation probabilities...")
    y_proba = pipeline.predict_proba(X_valid)[:, 1]

    auc = roc_auc_score(y_valid, y_proba)
    gini = 2.0 * auc - 1.0

    thresholds = np.round(
        np.arange(args.min_threshold, args.max_threshold + args.step, args.step),
        6,
    )

    metrics = evaluate_thresholds(y_valid, y_proba, thresholds)
    metrics.to_csv(REPORT_DIR / "threshold_metrics.csv", index=False)

    best_row = metrics.sort_values(["f1", "recall"], ascending=False).iloc[0].to_dict()

    best_summary = {
        "model": "xgboost",
        "selection_rule": "max_f1_then_recall",
        "auc": float(auc),
        "gini": float(gini),
        "best_threshold": float(best_row["threshold"]),
        "precision": float(best_row["precision"]),
        "recall": float(best_row["recall"]),
        "f1": float(best_row["f1"]),
        "tn": int(best_row["tn"]),
        "fp": int(best_row["fp"]),
        "fn": int(best_row["fn"]),
        "tp": int(best_row["tp"]),
        "predicted_default_rate": float(best_row["predicted_default_rate"]),
        "predicted_approval_rate": float(best_row["predicted_approval_rate"]),
        "false_positive_rate": float(best_row["false_positive_rate"]),
        "false_negative_rate": float(best_row["false_negative_rate"]),
    }

    with open(REPORT_DIR / "best_threshold.json", "w", encoding="utf-8") as f:
        json.dump(best_summary, f, indent=2)

    save_threshold_plots(metrics)

    model_path = MODEL_DIR / "final_xgboost_pipeline.joblib"
    joblib.dump(pipeline, model_path)

    print("\n" + "=" * 100)
    print("THRESHOLD TUNING SUMMARY")
    print("=" * 100)
    print(f"Validation AUC:  {auc:.6f}")
    print(f"Validation Gini: {gini:.6f}")
    print(f"Best threshold:  {best_summary['best_threshold']:.4f}")
    print(f"Precision:       {best_summary['precision']:.6f}")
    print(f"Recall:          {best_summary['recall']:.6f}")
    print(f"F1:              {best_summary['f1']:.6f}")
    print(f"TP / FP / FN / TN: {best_summary['tp']} / {best_summary['fp']} / {best_summary['fn']} / {best_summary['tn']}")

    print("\nSaved:")
    print("- models/final_xgboost_pipeline.joblib")
    print("- reports/threshold_metrics.csv")
    print("- reports/best_threshold.json")
    print("- reports/threshold_precision_recall.png")
    print("- reports/threshold_f1.png")
    print("- reports/threshold_confusion_counts.png")


if __name__ == "__main__":
    main()
