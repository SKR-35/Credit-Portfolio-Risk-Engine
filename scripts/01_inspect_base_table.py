from pathlib import Path
import polars as pl

BASE_FILE = Path(
    "data/raw/parquet_files/train/train_base.parquet"
)

df = pl.read_parquet(BASE_FILE)

print("=" * 80)
print("SHAPE")
print(df.shape)

print("=" * 80)
print("COLUMNS")
print(df.columns)

print("=" * 80)
print("DTYPES")
print(df.schema)

print("=" * 80)
print("TARGET DISTRIBUTION")

if "target" in df.columns:
    print(df["target"].value_counts())

print("=" * 80)
print("FIRST 5 ROWS")

print(df.head())

print("=" * 80)
print("MISSING VALUES")

missing = (
    df.null_count()
      .transpose(include_header=True)
      .rename({"column": "feature", "column_0": "null_count"})
      .sort("null_count", descending=True)
)

print(missing.head(30))