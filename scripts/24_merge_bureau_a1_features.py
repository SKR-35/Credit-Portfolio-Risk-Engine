from pathlib import Path
import polars as pl

BASE_PATH = Path(
    "data/processed/train_static_person_applprev_bureaub_features.parquet"
)

BUREAU_A1_PATH = Path(
    "data/processed/bureau_a1_features.parquet"
)

OUT_PATH = Path(
    "data/processed/train_static_person_applprev_bureaub_a1_features.parquet"
)

print("Loading datasets...")

base = pl.read_parquet(BASE_PATH)
bureau_a1 = pl.read_parquet(BUREAU_A1_PATH)

print("Base:", base.shape)
print("Bureau A1:", bureau_a1.shape)

print("Joining...")

df = base.join(
    bureau_a1,
    on="case_id",
    how="left"
)

print("Joined:", df.shape)

print("Filling bureau A1 features with 0...")

bureau_cols = [
    col for col in bureau_a1.columns
    if col != "case_id"
]

for col in bureau_cols:
    df = df.with_columns(
        pl.col(col).fill_null(0)
    )

df.write_parquet(OUT_PATH)

print(f"Saved -> {OUT_PATH}")