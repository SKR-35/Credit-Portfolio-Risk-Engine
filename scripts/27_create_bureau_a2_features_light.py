"""
Create lightweight Bureau A2 features for v0.1.

Context:
--------
Bureau A2 contains very large payment / collateral-level records.
On a local 8 GB RAM machine, full feature expansion would likely create
memory pressure during downstream model training.

For v0.1, we intentionally create a compact, high-signal Bureau A2 feature set.
The focus is on:

- payment days past due
- overdue payment amounts
- collateral / guarantee amounts
- record counts

In v1.0, this should be refactored into a chunked src/features/bureau.py pipeline.
"""

from pathlib import Path
import polars as pl

RAW_DIR = Path("data/raw/parquet_files/train")
OUT_PATH = Path("data/processed/bureau_a2_features_light.parquet")

OUT_PATH.parent.mkdir(
    parents=True,
    exist_ok=True
)

FILES = sorted(
    RAW_DIR.glob("train_credit_bureau_a_2_*.parquet")
)

RISK_COLS = [
    "pmts_dpd_1073P",
    "pmts_dpd_303P",
    "pmts_overdue_1140A",
    "pmts_overdue_1152A",
    "collater_valueofguarantee_1124L",
    "collater_valueofguarantee_876L",
]

partial_features = []

print("Creating lightweight Bureau A2 features...")

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
        pl.len().alias("bureau_a2_light_record_count")
    ]

    for col in available_cols:
        aggs.extend(
            [
                pl.col(col)
                .max()
                .alias(f"bureau_a2_light_{col}_max"),

                pl.col(col)
                .mean()
                .alias(f"bureau_a2_light_{col}_mean"),

                pl.col(col)
                .sum()
                .alias(f"bureau_a2_light_{col}_sum"),
            ]
        )

    part = (
        df
        .group_by("case_id")
        .agg(aggs)
    )

    print("Partial:", part.shape)

    partial_features.append(part)

print("Combining partial Bureau A2 features...")

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
    elif col.endswith("_mean"):
        final_aggs.append(
            pl.col(col).mean().alias(col)
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

print("Final lightweight Bureau A2 features:", features.shape)

features.write_parquet(
    OUT_PATH
)

print(f"Saved -> {OUT_PATH}")