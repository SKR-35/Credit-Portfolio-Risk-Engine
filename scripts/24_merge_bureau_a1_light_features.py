"""
Merge lightweight Bureau A1 features into the current v0.1 training dataset.

Context:
--------
The full Bureau A1 feature matrix was too wide for local XGBoost training
on an 8 GB RAM machine. For v0.1, we use a smaller Bureau A1 feature set
created by 23_create_bureau_a1_features_light.py.

Input:
------
data/processed/train_static_person_applprev_bureaub_features.parquet
data/processed/bureau_a1_features_light.parquet

Output:
-------
data/processed/train_static_person_applprev_bureaub_a1_light_features.parquet
"""

from pathlib import Path
import polars as pl

BASE_PATH = Path(
    "data/processed/train_static_person_applprev_bureaub_features.parquet"
)

BUREAU_A1_LIGHT_PATH = Path(
    "data/processed/bureau_a1_features_light.parquet"
)

OUT_PATH = Path(
    "data/processed/train_static_person_applprev_bureaub_a1_light_features.parquet"
)

print("Loading datasets...")

base = pl.read_parquet(BASE_PATH)
bureau_a1 = pl.read_parquet(BUREAU_A1_LIGHT_PATH)

print("Base:", base.shape)
print("Bureau A1 light:", bureau_a1.shape)

print("Joining...")

df = base.join(
    bureau_a1,
    on="case_id",
    how="left"
)

print("Joined:", df.shape)

print("Filling missing Bureau A1 light features with 0...")

bureau_cols = [
    col for col in bureau_a1.columns
    if col != "case_id"
]

for col in bureau_cols:
    if col in df.columns:
        df = df.with_columns(
            pl.col(col).fill_null(0)
        )

OUT_PATH.parent.mkdir(
    parents=True,
    exist_ok=True
)

df.write_parquet(OUT_PATH)

print(f"Saved -> {OUT_PATH}")