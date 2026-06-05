"""
10_optuna_xgboost_tuning.py

Real hyperparameter tuning for the selected XGBoost credit risk model.

What this script does:
1. Loads the processed static feature dataset.
2. Creates one fixed train/validation split.
3. Uses Optuna to search XGBoost hyperparameters.
4. Optimizes validation AUC.
5. Reports AUC, Gini, and Kaggle-style WEEK_NUM stability metric.
6. Saves:
   - tuned XGBoost sklearn Pipeline as joblib
   - best Optuna parameters
   - trial history
   - tuned model metrics
   - feature importance
   - optimization history plot, if optuna visualization dependencies are available

Usage:
    python scripts/10_optuna_xgboost_tuning.py

Recommended first run:
    python scripts/10_optuna_xgboost_tuning.py --sample 300000 --n-trials 20

Full run:
    python scripts/10_optuna_xgboost_tuning.py --n-trials 40

Notes:
- This can take time on a low-RAM laptop.
- If it is too slow, reduce --n-trials or use --sample.
- This script tunes the model, not only the threshold.
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

import optuna
from optuna.samplers import TPESampler

from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

from xgboost import XGBClassifier


warnings.filterwarnings("ignore")

DATA_PATH = Path("data/processed/train_static_features.parquet")
REPORT_DIR = Path("reports")
MODEL_DIR = Path("models")

REPORT_DIR.mkdir(parents=True, exist_ok=True)
MODEL_DIR.mkdir(parents=True, exist_ok=True)


def gini_from_auc(auc: float) -> float:
    return 2.0 * auc - 1.0


def stability_metric(
    y_true: pd.Series,
    y_pred: np.ndarray,
    week_num: pd.Series | None,
) -> dict:
    if week_num is None:
        return {
            "stability_metric": np.nan,
            "mean_weekly_gini": np.nan,
            "gini_slope": np.nan,
            "gini_residual_std": np.nan,
        }

    tmp = pd.DataFrame(
        {
            "target": y_true.to_numpy(),
            "prediction": y_pred,
            "week_num": week_num.to_numpy(),
        }
    )

    weekly = []
    for week, g in tmp.groupby("week_num"):
        if g["target"].nunique() < 2:
            continue

        auc = roc_auc_score(g["target"], g["prediction"])
        weekly.append(
            {
                "week_num": week,
                "auc": auc,
                "gini": gini_from_auc(auc),
                "n": len(g),
            }
        )

    weekly_df = pd.DataFrame(weekly).sort_values("week_num")

    if len(weekly_df) < 2:
        return {
            "stability_metric": np.nan,
            "mean_weekly_gini": np.nan,
            "gini_slope": np.nan,
            "gini_residual_std": np.nan,
        }

    x = weekly_df["week_num"].astype(float).to_numpy()
    y = weekly_df["gini"].astype(float).to_numpy()

    slope, intercept = np.polyfit(x, y, 1)
    fitted = slope * x + intercept
    residuals = y - fitted
    residual_std = float(np.std(residuals))

    metric = float(np.mean(y) + 88.0 * min(0.0, slope) - 0.5 * residual_std)

    weekly_df.to_csv(REPORT_DIR / "weekly_gini_tuned_xgboost.csv", index=False)

    return {
        "stability_metric": metric,
        "mean_weekly_gini": float(np.mean(y)),
        "gini_slope": float(slope),
        "gini_residual_std": residual_std,
    }


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


def make_xgb_model(params: dict, random_state: int) -> XGBClassifier:
    return XGBClassifier(
        objective="binary:logistic",
        eval_metric="auc",
        tree_method="hist",
        random_state=random_state,
        n_jobs=-1,
        **params,
    )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--sample", type=int, default=None)
    parser.add_argument("--n-trials", type=int, default=30)
    parser.add_argument("--random-state", type=int, default=42)
    parser.add_argument(
        "--timeout",
        type=int,
        default=None,
        help="Optional Optuna timeout in seconds.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    print("Loading dataset...")
    df = pl.read_parquet(DATA_PATH).to_pandas()
    print(f"Dataset shape: {df.shape}")

    if args.sample is not None and args.sample < len(df):
        print(f"Sampling {args.sample:,} rows for faster Optuna tuning...")
        df = (
            df.groupby("target", group_keys=False)
            .apply(lambda x: x.sample(frac=args.sample / len(df), random_state=args.random_state))
            .sample(frac=1.0, random_state=args.random_state)
            .reset_index(drop=True)
        )
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
        random_state=args.random_state,
        stratify=y,
    )

    week_valid_for_metric = None if week_num is None else week_valid

    preprocessor, numeric_cols, categorical_cols = build_preprocessor(X_train)

    print(f"Numeric features: {len(numeric_cols)}")
    print(f"Categorical features: {len(categorical_cols)}")
    print(f"Optuna trials: {args.n_trials}")

    def objective(trial: optuna.Trial) -> float:
        params = {
            "n_estimators": trial.suggest_int("n_estimators", 300, 1200, step=100),
            "learning_rate": trial.suggest_float("learning_rate", 0.01, 0.08, log=True),
            "max_depth": trial.suggest_int("max_depth", 3, 8),
            "min_child_weight": trial.suggest_float("min_child_weight", 1.0, 20.0, log=True),
            "subsample": trial.suggest_float("subsample", 0.65, 1.0),
            "colsample_bytree": trial.suggest_float("colsample_bytree", 0.65, 1.0),
            "gamma": trial.suggest_float("gamma", 0.0, 5.0),
            "reg_alpha": trial.suggest_float("reg_alpha", 1e-4, 5.0, log=True),
            "reg_lambda": trial.suggest_float("reg_lambda", 1e-3, 20.0, log=True),
            "max_delta_step": trial.suggest_int("max_delta_step", 0, 5),
        }

        model = make_xgb_model(params=params, random_state=args.random_state)

        pipeline = Pipeline(
            steps=[
                ("preprocessor", preprocessor),
                ("model", model),
            ]
        )

        pipeline.fit(X_train, y_train)
        preds = pipeline.predict_proba(X_valid)[:, 1]
        auc = roc_auc_score(y_valid, preds)

        trial.set_user_attr("gini", gini_from_auc(auc))

        return auc

    sampler = TPESampler(seed=args.random_state)
    study = optuna.create_study(
        direction="maximize",
        sampler=sampler,
        study_name="xgboost_credit_risk_tuning",
    )

    print("\nStarting Optuna tuning...")
    study.optimize(objective, n_trials=args.n_trials, timeout=args.timeout)

    trials_df = study.trials_dataframe()
    trials_df.to_csv(REPORT_DIR / "optuna_xgboost_trials.csv", index=False)

    best_params = study.best_params
    best_auc = study.best_value
    best_gini = gini_from_auc(best_auc)

    print("\nBest Optuna params:")
    print(json.dumps(best_params, indent=2))
    print(f"Best validation AUC from study: {best_auc:.6f}")
    print(f"Best validation Gini from study: {best_gini:.6f}")

    print("\nTraining final tuned XGBoost pipeline...")
    final_model = make_xgb_model(params=best_params, random_state=args.random_state)

    final_pipeline = Pipeline(
        steps=[
            ("preprocessor", preprocessor),
            ("model", final_model),
        ]
    )

    final_pipeline.fit(X_train, y_train)
    final_preds = final_pipeline.predict_proba(X_valid)[:, 1]

    final_auc = roc_auc_score(y_valid, final_preds)
    final_gini = gini_from_auc(final_auc)
    stability = stability_metric(y_valid, final_preds, week_valid_for_metric)

    final_summary = {
        "model": "tuned_xgboost",
        "tuning_method": "optuna_tpe",
        "n_trials": args.n_trials,
        "sample": args.sample,
        "auc": float(final_auc),
        "gini": float(final_gini),
        "stability_metric": stability["stability_metric"],
        "mean_weekly_gini": stability["mean_weekly_gini"],
        "gini_slope": stability["gini_slope"],
        "gini_residual_std": stability["gini_residual_std"],
        "best_params": best_params,
    }

    with open(REPORT_DIR / "tuned_xgboost_summary.json", "w", encoding="utf-8") as f:
        json.dump(final_summary, f, indent=2)

    with open(REPORT_DIR / "tuned_xgboost_best_params.json", "w", encoding="utf-8") as f:
        json.dump(best_params, f, indent=2)

    joblib.dump(final_pipeline, MODEL_DIR / "tuned_xgboost_pipeline.joblib")

    # Feature importance after preprocessing.
    try:
        transformed_feature_names = final_pipeline.named_steps["preprocessor"].get_feature_names_out()
        importances = final_pipeline.named_steps["model"].feature_importances_

        importance_df = pd.DataFrame(
            {
                "feature": transformed_feature_names,
                "importance": importances,
            }
        ).sort_values("importance", ascending=False)

        importance_df.to_csv(REPORT_DIR / "feature_importance_tuned_xgboost.csv", index=False)
    except Exception as exc:
        print(f"Could not save transformed feature importance: {exc}")

    # Simple Optuna history plot via matplotlib.
    try:
        import matplotlib.pyplot as plt

        complete_trials = [
            t for t in study.trials
            if t.value is not None and t.state.name == "COMPLETE"
        ]
        trial_numbers = [t.number for t in complete_trials]
        values = [t.value for t in complete_trials]
        best_so_far = np.maximum.accumulate(values)

        plt.figure(figsize=(9, 6))
        plt.plot(trial_numbers, values, marker="o", label="Trial AUC")
        plt.plot(trial_numbers, best_so_far, marker="o", label="Best so far")
        plt.xlabel("Trial")
        plt.ylabel("Validation AUC")
        plt.title("Optuna XGBoost Tuning History")
        plt.legend()
        plt.tight_layout()
        plt.savefig(REPORT_DIR / "optuna_xgboost_history.png", dpi=150)
        plt.close()
    except Exception as exc:
        print(f"Could not save Optuna history plot: {exc}")

    print("\n" + "=" * 100)
    print("OPTUNA XGBOOST TUNING SUMMARY")
    print("=" * 100)
    print(f"Final tuned validation AUC:  {final_auc:.6f}")
    print(f"Final tuned validation Gini: {final_gini:.6f}")
    print(f"Stability metric:            {final_summary['stability_metric']:.6f}")
    print(f"Mean weekly Gini:            {final_summary['mean_weekly_gini']:.6f}")
    print(f"Gini slope:                  {final_summary['gini_slope']:.6f}")
    print(f"Gini residual std:           {final_summary['gini_residual_std']:.6f}")

    print("\nSaved:")
    print("- models/tuned_xgboost_pipeline.joblib")
    print("- reports/tuned_xgboost_summary.json")
    print("- reports/tuned_xgboost_best_params.json")
    print("- reports/optuna_xgboost_trials.csv")
    print("- reports/optuna_xgboost_history.png")
    print("- reports/feature_importance_tuned_xgboost.csv")
    print("- reports/weekly_gini_tuned_xgboost.csv")


if __name__ == "__main__":
    main()
