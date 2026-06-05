"""
08_compare_models.py

Compare multiple credit risk models with:
- AUC
- Gini = 2 * AUC - 1
- Kaggle-style weekly stability metric when WEEK_NUM exists
- Precision / Recall / F1 at a configurable threshold
- Model comparison CSV
- ROC curve artifact

Usage:
    python scripts/08_compare_models.py
    python scripts/08_compare_models.py --sample 300000
    python scripts/08_compare_models.py --skip-logistic
    python scripts/08_compare_models.py --skip-random-forest
    python scripts/08_compare_models.py --include-xgboost
"""

from __future__ import annotations

import argparse
import json
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
import polars as pl
import lightgbm as lgb

from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    roc_auc_score,
    precision_score,
    recall_score,
    f1_score,
    confusion_matrix,
    roc_curve,
)
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

try:
    from xgboost import XGBClassifier
    HAS_XGBOOST = True
except Exception:
    HAS_XGBOOST = False

warnings.filterwarnings("ignore")

DATA_PATH = Path("data/processed/train_static_features.parquet")
REPORT_DIR = Path("reports")
MODEL_DIR = Path("models")
REPORT_DIR.mkdir(parents=True, exist_ok=True)
MODEL_DIR.mkdir(parents=True, exist_ok=True)


def gini_from_auc(auc: float) -> float:
    return 2.0 * auc - 1.0


def stability_metric(y_true: pd.Series, y_pred: np.ndarray, week_num: pd.Series) -> dict:
    tmp = pd.DataFrame({
        "target": y_true.to_numpy(),
        "prediction": y_pred,
        "week_num": week_num.to_numpy(),
    })

    weekly = []
    for week, g in tmp.groupby("week_num"):
        if g["target"].nunique() < 2:
            continue
        auc = roc_auc_score(g["target"], g["prediction"])
        weekly.append({"week_num": week, "auc": auc, "gini": gini_from_auc(auc), "n": len(g)})

    weekly_df = pd.DataFrame(weekly).sort_values("week_num")

    if len(weekly_df) < 2:
        return {
            "stability_metric": np.nan,
            "mean_gini": np.nan,
            "slope": np.nan,
            "residual_std": np.nan,
            "weekly_scores": weekly_df,
        }

    x = weekly_df["week_num"].astype(float).to_numpy()
    y = weekly_df["gini"].astype(float).to_numpy()
    slope, intercept = np.polyfit(x, y, 1)
    fitted = slope * x + intercept
    residuals = y - fitted
    residual_std = float(np.std(residuals))

    metric = float(np.mean(y) + 88.0 * min(0.0, slope) - 0.5 * residual_std)

    return {
        "stability_metric": metric,
        "mean_gini": float(np.mean(y)),
        "slope": float(slope),
        "residual_std": residual_std,
        "weekly_scores": weekly_df,
    }


def evaluate_predictions(model_name, y_true, y_pred_proba, threshold, week_num=None):
    y_pred_label = (y_pred_proba >= threshold).astype(int)

    auc = roc_auc_score(y_true, y_pred_proba)
    gini = gini_from_auc(auc)
    precision = precision_score(y_true, y_pred_label, zero_division=0)
    recall = recall_score(y_true, y_pred_label, zero_division=0)
    f1 = f1_score(y_true, y_pred_label, zero_division=0)
    tn, fp, fn, tp = confusion_matrix(y_true, y_pred_label).ravel()

    result = {
        "model": model_name,
        "auc": auc,
        "gini": gini,
        "precision": precision,
        "recall": recall,
        "f1": f1,
        "threshold": threshold,
        "tn": int(tn),
        "fp": int(fp),
        "fn": int(fn),
        "tp": int(tp),
    }

    if week_num is not None:
        stability = stability_metric(y_true, y_pred_proba, week_num)
        result.update({
            "stability_metric": stability["stability_metric"],
            "mean_weekly_gini": stability["mean_gini"],
            "gini_slope": stability["slope"],
            "gini_residual_std": stability["residual_std"],
        })
        stability["weekly_scores"].to_csv(REPORT_DIR / f"weekly_gini_{model_name}.csv", index=False)

    return result


def build_preprocessor(X: pd.DataFrame):
    categorical_cols = [
        col for col in X.columns
        if X[col].dtype == "object" or str(X[col].dtype) == "category"
    ]
    numeric_cols = [col for col in X.columns if col not in categorical_cols]

    numeric_pipe = Pipeline([
        ("imputer", SimpleImputer(strategy="median")),
        ("scaler", StandardScaler(with_mean=False)),
    ])

    categorical_pipe = Pipeline([
        ("imputer", SimpleImputer(strategy="most_frequent")),
        ("onehot", OneHotEncoder(handle_unknown="ignore", min_frequency=50, sparse_output=True)),
    ])

    preprocessor = ColumnTransformer(
        [
            ("num", numeric_pipe, numeric_cols),
            ("cat", categorical_pipe, categorical_cols),
        ],
        remainder="drop",
        sparse_threshold=0.3,
    )

    return preprocessor, numeric_cols, categorical_cols


def train_lightgbm_native(X_train, X_valid, y_train, y_valid, categorical_cols):
    X_train_lgb = X_train.copy()
    X_valid_lgb = X_valid.copy()

    for col in categorical_cols:
        X_train_lgb[col] = X_train_lgb[col].astype("category")
        X_valid_lgb[col] = X_valid_lgb[col].astype("category")

    for col in X_train_lgb.columns:
        if str(X_train_lgb[col].dtype) == "bool":
            X_train_lgb[col] = X_train_lgb[col].astype("int8")
            X_valid_lgb[col] = X_valid_lgb[col].astype("int8")

    model = lgb.LGBMClassifier(
        objective="binary",
        n_estimators=800,
        learning_rate=0.03,
        num_leaves=64,
        subsample=0.85,
        colsample_bytree=0.85,
        reg_alpha=0.1,
        reg_lambda=1.0,
        class_weight="balanced",
        random_state=42,
        n_jobs=-1,
    )

    model.fit(
        X_train_lgb,
        y_train,
        categorical_feature=categorical_cols,
        eval_set=[(X_valid_lgb, y_valid)],
        eval_metric="auc",
        callbacks=[lgb.early_stopping(50), lgb.log_evaluation(100)],
    )

    preds = model.predict_proba(X_valid_lgb)[:, 1]
    model.booster_.save_model(str(MODEL_DIR / "compare_lightgbm.txt"))

    importance = pd.DataFrame({
        "feature": X_train_lgb.columns,
        "importance": model.feature_importances_,
    }).sort_values("importance", ascending=False)
    importance.to_csv(REPORT_DIR / "feature_importance_compare_lightgbm.csv", index=False)

    return model, preds


def train_pipeline_model(model_name, estimator, preprocessor, X_train, X_valid, y_train):
    pipe = Pipeline([("preprocessor", preprocessor), ("model", estimator)])
    pipe.fit(X_train, y_train)
    preds = pipe.predict_proba(X_valid)[:, 1]
    return pipe, preds


def save_roc_curves(y_valid, prediction_map):
    import matplotlib.pyplot as plt

    plt.figure(figsize=(8, 6))
    for model_name, preds in prediction_map.items():
        fpr, tpr, _ = roc_curve(y_valid, preds)
        auc = roc_auc_score(y_valid, preds)
        plt.plot(fpr, tpr, label=f"{model_name} AUC={auc:.4f}")

    plt.plot([0, 1], [0, 1], linestyle="--", label="Random")
    plt.xlabel("False Positive Rate")
    plt.ylabel("True Positive Rate")
    plt.title("ROC Curve Comparison")
    plt.legend()
    plt.tight_layout()
    plt.savefig(REPORT_DIR / "roc_curve_model_comparison.png", dpi=150)
    plt.close()


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--sample", type=int, default=None)
    parser.add_argument("--threshold", type=float, default=0.5)
    parser.add_argument(
        "--skip-logistic",
        action="store_true",
        help=(
            "Skip Logistic Regression. Useful for full-data runs on low-RAM machines, "
            "because the sklearn preprocessing pipeline may allocate large float64 arrays."
        ),
    )
    parser.add_argument("--skip-random-forest", action="store_true")
    parser.add_argument("--include-xgboost", action="store_true")
    return parser.parse_args()


def main():
    args = parse_args()

    print("Loading dataset...")
    df = pl.read_parquet(DATA_PATH).to_pandas()
    print(f"Dataset shape: {df.shape}")

    if args.sample is not None and args.sample < len(df):
        print(f"Sampling {args.sample:,} rows for faster comparison...")
        df = df.groupby("target", group_keys=False).apply(
            lambda x: x.sample(frac=args.sample / len(df), random_state=42)
        ).sample(frac=1.0, random_state=42).reset_index(drop=True)
        print(f"Sampled shape: {df.shape}")

    target_col = "target"
    drop_cols = ["case_id", "target"]
    week_num = df["WEEK_NUM"] if "WEEK_NUM" in df.columns else None

    X = df.drop(columns=drop_cols)
    y = df[target_col]

    X_train, X_valid, y_train, y_valid, week_train, week_valid = train_test_split(
        X,
        y,
        week_num if week_num is not None else pd.Series(np.nan, index=df.index),
        test_size=0.2,
        random_state=42,
        stratify=y,
    )

    week_valid_for_metric = None if week_num is None else week_valid
    preprocessor, numeric_cols, categorical_cols = build_preprocessor(X_train)

    print(f"Numeric features: {len(numeric_cols)}")
    print(f"Categorical features: {len(categorical_cols)}")

    results = []
    prediction_map = {}

    if args.skip_logistic:
        print("\nSkipping Logistic Regression...")
        print(
            "Reason: full-data Logistic Regression can exceed RAM on low-memory machines. "
            "The sklearn preprocessing pipeline may create large temporary float64 arrays."
        )
    else:
        print("\nTraining Logistic Regression...")
        # NOTE:
        # On the full 1.5M-row dataset, Logistic Regression can fail on low-RAM machines.
        # The sklearn preprocessing pipeline may allocate large temporary float64 arrays
        # during imputation/scaling/OHE. Use --skip-logistic for full-data runs.
        logistic = LogisticRegression(
            max_iter=1000,
            class_weight="balanced",
            solver="saga",
            n_jobs=-1,
            random_state=42,
        )
        _, preds_logistic = train_pipeline_model(
            "logistic_regression", logistic, preprocessor, X_train, X_valid, y_train
        )
        prediction_map["logistic_regression"] = preds_logistic
        results.append(evaluate_predictions(
            "logistic_regression", y_valid, preds_logistic, args.threshold, week_valid_for_metric
        ))

    if not args.skip_random_forest:
        print("\nTraining Random Forest...")
        rf = RandomForestClassifier(
            n_estimators=250,
            max_depth=14,
            min_samples_leaf=50,
            class_weight="balanced_subsample",
            random_state=42,
            n_jobs=-1,
        )
        _, preds_rf = train_pipeline_model(
            "random_forest", rf, preprocessor, X_train, X_valid, y_train
        )
        prediction_map["random_forest"] = preds_rf
        results.append(evaluate_predictions(
            "random_forest", y_valid, preds_rf, args.threshold, week_valid_for_metric
        ))

    print("\nTraining LightGBM native categorical model...")
    _, preds_lgbm = train_lightgbm_native(
        X_train, X_valid, y_train, y_valid, categorical_cols
    )
    prediction_map["lightgbm_native"] = preds_lgbm
    results.append(evaluate_predictions(
        "lightgbm_native", y_valid, preds_lgbm, args.threshold, week_valid_for_metric
    ))

    if args.include_xgboost:
        if not HAS_XGBOOST:
            print("\nXGBoost requested but not installed. Skipping.")
        else:
            print("\nTraining XGBoost...")
            xgb = XGBClassifier(
                n_estimators=600,
                learning_rate=0.03,
                max_depth=6,
                subsample=0.85,
                colsample_bytree=0.85,
                objective="binary:logistic",
                eval_metric="auc",
                tree_method="hist",
                random_state=42,
                n_jobs=-1,
            )
            _, preds_xgb = train_pipeline_model(
                "xgboost", xgb, preprocessor, X_train, X_valid, y_train
            )
            prediction_map["xgboost"] = preds_xgb
            results.append(evaluate_predictions(
                "xgboost", y_valid, preds_xgb, args.threshold, week_valid_for_metric
            ))

    results_df = pd.DataFrame(results).sort_values("auc", ascending=False)
    results_df.to_csv(REPORT_DIR / "model_comparison.csv", index=False)
    save_roc_curves(y_valid, prediction_map)

    best = results_df.iloc[0].to_dict()
    with open(REPORT_DIR / "best_model_summary.json", "w", encoding="utf-8") as f:
        json.dump(best, f, indent=2)

    print("\n" + "=" * 100)
    print("MODEL COMPARISON")
    print("=" * 100)
    print(results_df.to_string(index=False))
    print("\nSaved:")
    print("- reports/model_comparison.csv")
    print("- reports/roc_curve_model_comparison.png")
    print("- reports/best_model_summary.json")
    print("- reports/weekly_gini_<model>.csv, if WEEK_NUM exists")
    print("- models/compare_lightgbm.txt")


if __name__ == "__main__":
    main()
