from pathlib import Path

import lightgbm as lgb
import polars as pl
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import train_test_split

DATA_PATH = Path("data/processed/base_features.parquet")
MODEL_DIR = Path("models")
MODEL_DIR.mkdir(parents=True, exist_ok=True)

df = pl.read_parquet(DATA_PATH).to_pandas()

target_col = "target"
id_col = "case_id"

X = df.drop(columns=[target_col, id_col])
y = df[target_col]

X_train, X_valid, y_train, y_valid = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42,
    stratify=y,
)

model = lgb.LGBMClassifier(
    objective="binary",
    n_estimators=300,
    learning_rate=0.05,
    num_leaves=31,
    class_weight="balanced",
    random_state=42,
    n_jobs=-1,
)

model.fit(
    X_train,
    y_train,
    eval_set=[(X_valid, y_valid)],
    eval_metric="auc",
)

preds = model.predict_proba(X_valid)[:, 1]
auc = roc_auc_score(y_valid, preds)

print("=" * 80)
print(f"Validation AUC: {auc:.6f}")
print("=" * 80)

model.booster_.save_model(str(MODEL_DIR / "base_lgbm.txt"))
print("Saved model to models/base_lgbm.txt")