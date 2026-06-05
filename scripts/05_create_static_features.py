from pathlib import Path
import polars as pl

BASE_PATH = Path(
    "data/processed/base_features.parquet"
)

STATIC_0 = Path(
    "data/raw/parquet_files/train/train_static_0_0.parquet"
)

STATIC_1 = Path(
    "data/raw/parquet_files/train/train_static_0_1.parquet"
)

OUT_PATH = Path(
    "data/processed/train_static_features.parquet"
)

print("Loading datasets...")

base = pl.read_parquet(BASE_PATH)

static_0 = pl.read_parquet(STATIC_0)
static_1 = pl.read_parquet(STATIC_1)

print("Concatenating static tables...")

static = pl.concat(
    [static_0, static_1],
    how="vertical"
)

print(static.shape)

print("Joining with base...")

df = base.join(
    static,
    on="case_id",
    how="left"
)

print(df.shape)

print("Dropping high missing columns...")

threshold = 0.95

null_stats = (
    df.null_count()
    .transpose(include_header=True)
    .rename(
        {
            "column": "feature",
            "column_0": "null_count"
        }
    )
)

cols_to_drop = []

for row in null_stats.iter_rows(named=True):

    ratio = row["null_count"] / df.height

    if ratio > threshold:
        cols_to_drop.append(row["feature"])

print(f"Columns removed: {len(cols_to_drop)}")

df = df.drop(cols_to_drop)

print(df.shape)

OUT_PATH.parent.mkdir(
    parents=True,
    exist_ok=True
)

df.write_parquet(OUT_PATH)

print(f"Saved -> {OUT_PATH}")