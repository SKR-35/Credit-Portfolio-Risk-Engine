from pathlib import Path
import polars as pl

BASE_PATH = Path(
    "data/processed/train_static_person_applprev_features.parquet"
)

BUREAU_B_PATH = Path(
    "data/processed/bureau_b_features.parquet"
)

OUT_PATH = Path(
    "data/processed/train_static_person_applprev_bureaub_features.parquet"
)

print("Loading datasets...")

base = pl.read_parquet(BASE_PATH)
bureau_b = pl.read_parquet(BUREAU_B_PATH)

print("Base:", base.shape)
print("Bureau B:", bureau_b.shape)

print("Joining...")

df = base.join(
    bureau_b,
    on="case_id",
    how="left"
)

print("Joined:", df.shape)

print("Filling missing bureau B features with 0...")

bureau_cols = [
    col for col in bureau_b.columns
    if col != "case_id"
]

for col in bureau_cols:
    if col in df.columns:
        df = df.with_columns(
            pl.col(col).fill_null(0)
        )

OUT_PATH.parent.mkdir(parents=True, exist_ok=True)
df.write_parquet(OUT_PATH)

print(f"Saved -> {OUT_PATH}")