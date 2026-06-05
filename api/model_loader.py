from pathlib import Path
import joblib

MODEL_PATH = Path("models/final_v01_xgboost.joblib")

model = joblib.load(MODEL_PATH)

FEATURE_NAMES = list(model.get_booster().feature_names)