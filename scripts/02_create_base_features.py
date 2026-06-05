from pathlib import Path
import polars as pl

RAW_PATH = Path("data/raw/parquet_files/train/train_base.parquet")
OUT_PATH = Path("data/processed/base_features.parquet")

OUT_PATH.parent.mkdir(parents=True, exist_ok=True)

df = pl.read_parquet(RAW_PATH)

df = df.with_columns(
    pl.col("date_decision").str.strptime(pl.Date, "%Y-%m-%d").alias("date_decision")
)

df = df.with_columns(
    [
        pl.col("date_decision").dt.year().alias("decision_year"),
        pl.col("date_decision").dt.month().alias("decision_month"),
        pl.col("date_decision").dt.weekday().alias("decision_weekday"),
    ]
)

df = df.drop("date_decision")

df.write_parquet(OUT_PATH)

print(f"Saved: {OUT_PATH}")
print(df.shape)
print(df.head())