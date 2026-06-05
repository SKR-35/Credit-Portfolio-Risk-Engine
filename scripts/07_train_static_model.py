from pathlib import Path
import pandas as pd
import polars as pl
import lightgbm as lgb

from sklearn.metrics import roc_auc_score
from sklearn.model_selection import train_test_split

DATA_PATH = Path(
    "data/processed/train_static_features.parquet"
)

MODEL_DIR = Path("models")
MODEL_DIR.mkdir(exist_ok=True)

print("Loading dataset...")

df = pl.read_parquet(DATA_PATH)

print(df.shape)

pdf = df.to_pandas()

target_col = "target"

drop_cols = [
    "case_id",
    "target"
]

X = pdf.drop(columns=drop_cols)
y = pdf[target_col]

print("Preparing categorical features...")

categorical_cols = []

for col in X.columns:

    if X[col].dtype == "object":

        X[col] = X[col].astype("category")
        categorical_cols.append(col)

    elif str(X[col].dtype) == "bool":

        X[col] = X[col].astype("int8")

print(
    f"Categorical features: {len(categorical_cols)}"
)

X_train, X_valid, y_train, y_valid = train_test_split(
    X,
    y,
    test_size=0.2,
    stratify=y,
    random_state=42
)

print("Training LightGBM...")

model = lgb.LGBMClassifier(
    objective="binary",
    n_estimators=500,
    learning_rate=0.05,
    num_leaves=64,
    class_weight="balanced",
    random_state=42,
    n_jobs=-1
)

model.fit(
    X_train,
    y_train,
    categorical_feature=categorical_cols
)

preds = model.predict_proba(X_valid)[:, 1]

auc = roc_auc_score(
    y_valid,
    preds
)

print("=" * 80)
print(f"AUC: {auc:.6f}")
print("=" * 80)

importance = pd.DataFrame(
    {
        "feature": X.columns,
        "importance": model.feature_importances_
    }
)

importance = (
    importance
    .sort_values(
        "importance",
        ascending=False
    )
)

print("\nTOP 20 FEATURES\n")
print(
    importance.head(20)
)

importance.to_csv(
    "reports/feature_importance_static.csv",
    index=False
)

model.booster_.save_model(
    "models/static_lgbm.txt"
)

print("\nModel saved.")
print("Feature importance saved.")