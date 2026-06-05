from pathlib import Path
import json
import mlflow

MODEL_PATH = Path("models/final_v01_xgboost.joblib")
SUMMARY_PATH = Path("reports/final_v01_xgboost_summary.json")
IMPORTANCE_PATH = Path("reports/feature_importance_final_v01_xgboost.csv")

mlflow.set_tracking_uri("sqlite:///mlflow_v01.db")
mlflow.set_experiment("credit-portfolio-risk-v01")


with open(SUMMARY_PATH, "r", encoding="utf-8") as f:
    summary = json.load(f)

with mlflow.start_run(run_name="final_v01_xgboost"):

    mlflow.log_param("model", summary.get("model", "XGBoost"))
    mlflow.log_param("version", summary.get("version", "v0.1"))
    mlflow.log_param("dataset", summary.get("dataset"))
    mlflow.log_param("rows", summary.get("rows"))
    mlflow.log_param("columns", summary.get("columns"))
    mlflow.log_param("encoded_categorical_features", summary.get("encoded_categorical_features"))
    mlflow.log_param("scale_pos_weight", summary.get("scale_pos_weight"))

    mlflow.log_metric("validation_auc", summary["validation_auc"])
    mlflow.log_metric("validation_gini", summary["validation_gini"])

    mlflow.log_artifact(str(MODEL_PATH), artifact_path="model")
    mlflow.log_artifact(str(SUMMARY_PATH), artifact_path="reports")
    mlflow.log_artifact(str(IMPORTANCE_PATH), artifact_path="reports")

print("Logged final v0.1 model to MLflow.")