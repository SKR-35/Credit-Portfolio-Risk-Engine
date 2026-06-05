from fastapi import FastAPI
import pandas as pd

from api.schemas import PredictionRequest
from api.model_loader import model, FEATURE_NAMES

app = FastAPI(
    title="Credit Portfolio Risk Engine",
    version="0.1"
)


@app.get("/")
def health_check():
    return {
        "status": "ok",
        "model": "final_v01_xgboost",
        "n_features": len(FEATURE_NAMES),
    }


@app.post("/predict")
def predict(request: PredictionRequest):
    row = {feature: 0.0 for feature in FEATURE_NAMES}

    for key, value in request.features.items():
        if key in row:
            row[key] = value

    data = pd.DataFrame([row], columns=FEATURE_NAMES)

    probability = model.predict_proba(data)[0, 1]

    return {
        "default_probability": float(probability),
        "risk_band": (
            "high" if probability >= 0.5
            else "medium" if probability >= 0.2
            else "low"
        ),
        "features_received": len(request.features),
        "model_features": len(FEATURE_NAMES),
    }