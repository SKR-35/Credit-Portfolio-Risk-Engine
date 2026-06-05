"""
Create lightweight Bureau A1 features for v0.1.

Context:
--------
The full Bureau A1 feature aggregation produced a wide feature matrix.
On an 8 GB RAM local machine, the downstream XGBoost training step failed
during DMatrix allocation because the combined dataset became too large.

For v0.1, we intentionally use a smaller, high-signal Bureau A1 feature set.
The goal is to keep the pipeline runnable locally while still capturing the
most important credit-risk signals:

- outstanding debt
- overdue debt
- maximum DPD
- overdue installment counts
- overdue amounts
- total outstanding debt

In v1.0, this should be refactored into a proper chunked feature pipeline
under src/features/, with stronger typing, feature selection, and model
comparison after the full feature matrix is created.
"""

from pathlib import Path
import polars as pl

RAW_DIR = Path("data/raw/parquet_files/train")

OUT_PATH = Path(
    "data/processed/bureau_a1_features_light.parquet"
)

OUT_PATH.parent.mkdir(
    parents=True,
    exist_ok=True
)

FILES = sorted(
    RAW_DIR.glob("train_credit_bureau_a_1_*.parquet")
)

RISK_COLS = [
    "debtoutstand_525A",
    "debtoverdue_47A",
    "dpdmax_139P",
    "dpdmax_757P",
    "numberofoverdueinstlmax_1039L",
    "numberofoverdueinstlmax_1151L",
    "numberofoverdueinstls_725L",
    "numberofoverdueinstls_834L",
    "overdueamount_31A",
    "overdueamount_659A",
    "overdueamountmax_155A",
    "overdueamountmax_35A",
    "totaldebtoverduevalue_178A",
    "totaldebtoverduevalue_718A",
    "totaloutstanddebtvalue_39A",
    "totaloutstanddebtvalue_668A",
]

print("Creating lightweight Bureau A1 features...")

partial_features = []

for file in FILES:
    print(f"Processing {file.name}")

    df = pl.read_parquet(file)

    available_cols = [
        col for col in RISK_COLS
        if col in df.columns
    ]

    df = df.select(
        ["case_id"] + available_cols
    )

    aggs = [
        pl.len().alias("bureau_a1_light_record_count")
    ]

    for col in available_cols:
        aggs.extend(
            [
                pl.col(col)
                .max()
                .alias(f"bureau_a1_light_{col}_max"),

                pl.col(col)
                .sum()
                .alias(f"bureau_a1_light_{col}_sum"),
            ]
        )

    part = (
        df
        .group_by("case_id")
        .agg(aggs)
    )

    print("Partial:", part.shape)

    partial_features.append(part)

print("Combining partial Bureau A1 features...")

combined = pl.concat(
    partial_features,
    how="diagonal_relaxed"
)

feature_cols = [
    col for col in combined.columns
    if col != "case_id"
]

final_aggs = []

for col in feature_cols:
    if col.endswith("_record_count"):
        final_aggs.append(
            pl.col(col).sum().alias(col)
        )
    elif col.endswith("_max"):
        final_aggs.append(
            pl.col(col).max().alias(col)
        )
    elif col.endswith("_sum"):
        final_aggs.append(
            pl.col(col).sum().alias(col)
        )

features = (
    combined
    .group_by("case_id")
    .agg(final_aggs)
)

print("Final lightweight Bureau A1 features:", features.shape)

features.write_parquet(
    OUT_PATH
)

print(f"Saved -> {OUT_PATH}")