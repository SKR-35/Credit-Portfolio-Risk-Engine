from pathlib import Path
import polars as pl

RAW_DIR = Path("data/raw/parquet_files/train")
OUT_PATH = Path("data/processed/bureau_a1_features.parquet")
OUT_PATH.parent.mkdir(parents=True, exist_ok=True)

FILES = sorted(RAW_DIR.glob("train_credit_bureau_a_1_*.parquet"))

RISK_COLS = [
    "debtoutstand_525A",
    "debtoverdue_47A",
    "dpdmax_139P",
    "dpdmax_757P",
    "monthlyinstlamount_332A",
    "monthlyinstlamount_674A",
    "numberofinstls_229L",
    "numberofinstls_320L",
    "numberofoutstandinstls_520L",
    "numberofoutstandinstls_59L",
    "numberofoverdueinstlmax_1039L",
    "numberofoverdueinstlmax_1151L",
    "numberofoverdueinstls_725L",
    "numberofoverdueinstls_834L",
    "outstandingamount_354A",
    "outstandingamount_362A",
    "overdueamount_31A",
    "overdueamount_659A",
    "overdueamountmax_155A",
    "overdueamountmax_35A",
    "residualamount_488A",
    "residualamount_856A",
    "totalamount_6A",
    "totalamount_996A",
    "totaldebtoverduevalue_178A",
    "totaldebtoverduevalue_718A",
    "totaloutstanddebtvalue_39A",
    "totaloutstanddebtvalue_668A",
]

print("Loading and aggregating Bureau A1 files...")

partial_features = []

for file in FILES:
    print(f"Processing {file.name}")

    df = pl.read_parquet(file)

    available_cols = [
        col for col in RISK_COLS
        if col in df.columns
    ]

    aggs = [
        pl.len().alias("bureau_a1_record_count")
    ]

    for col in available_cols:
        aggs.extend(
            [
                pl.col(col).max().alias(f"bureau_a1_{col}_max"),
                pl.col(col).mean().alias(f"bureau_a1_{col}_mean"),
                pl.col(col).sum().alias(f"bureau_a1_{col}_sum"),
            ]
        )

    part = (
        df
        .select(["case_id"] + available_cols)
        .group_by("case_id")
        .agg(aggs)
    )

    print("Partial:", part.shape)
    partial_features.append(part)

print("Combining partial aggregates...")

combined = pl.concat(
    partial_features,
    how="diagonal_relaxed"
)

print("Combined partial:", combined.shape)

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
    elif col.endswith("_mean"):
        final_aggs.append(
            pl.col(col).mean().alias(col)
        )

features = (
    combined
    .group_by("case_id")
    .agg(final_aggs)
)

print("Final Bureau A1 features:", features.shape)

features.write_parquet(OUT_PATH)

print(f"Saved -> {OUT_PATH}")