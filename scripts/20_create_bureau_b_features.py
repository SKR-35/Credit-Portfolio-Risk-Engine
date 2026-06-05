from pathlib import Path
import polars as pl

B1_PATH = Path("data/raw/parquet_files/train/train_credit_bureau_b_1.parquet")
B2_PATH = Path("data/raw/parquet_files/train/train_credit_bureau_b_2.parquet")

OUT_PATH = Path("data/processed/bureau_b_features.parquet")
OUT_PATH.parent.mkdir(parents=True, exist_ok=True)

print("Loading bureau B tables...")

b1 = pl.read_parquet(B1_PATH)
b2 = pl.read_parquet(B2_PATH)

print("B1:", b1.shape)
print("B2:", b2.shape)

print("Creating B1 aggregates...")

b1_features = (
    b1.group_by("case_id")
    .agg(
        [
            pl.len().alias("bureau_b1_count"),

            pl.col("dpd_550P").max().alias("bureau_b1_dpd_550_max"),
            pl.col("dpd_550P").mean().alias("bureau_b1_dpd_550_mean"),

            pl.col("dpd_733P").max().alias("bureau_b1_dpd_733_max"),
            pl.col("dpd_733P").mean().alias("bureau_b1_dpd_733_mean"),

            pl.col("dpdmax_851P").max().alias("bureau_b1_dpdmax_max"),
            pl.col("dpdmax_851P").mean().alias("bureau_b1_dpdmax_mean"),

            pl.col("debtpastduevalue_732A")
            .max()
            .alias("bureau_b1_debtpastdue_max"),

            pl.col("debtpastduevalue_732A")
            .sum()
            .alias("bureau_b1_debtpastdue_sum"),

            pl.col("debtvalue_227A")
            .max()
            .alias("bureau_b1_debtvalue_max"),

            pl.col("debtvalue_227A")
            .sum()
            .alias("bureau_b1_debtvalue_sum"),

            pl.col("overdueamountmax_950A")
            .max()
            .alias("bureau_b1_overdueamountmax_max"),

            pl.col("maxdebtpduevalodued_3940955A")
            .max()
            .alias("bureau_b1_maxdebtpastdue_max"),

            pl.col("credlmt_1052A")
            .max()
            .alias("bureau_b1_credlmt_1052_max"),

            pl.col("credlmt_228A")
            .max()
            .alias("bureau_b1_credlmt_228_max"),

            pl.col("totalamount_503A")
            .sum()
            .alias("bureau_b1_totalamount_503_sum"),

            pl.col("totalamount_881A")
            .sum()
            .alias("bureau_b1_totalamount_881_sum"),

            pl.col("numberofinstls_810L")
            .max()
            .alias("bureau_b1_numberofinstls_max"),

            pl.col("pmtnumpending_403L")
            .max()
            .alias("bureau_b1_pmtnumpending_max"),
        ]
    )
)

print("Creating B2 aggregates...")

b2_features = (
    b2.group_by("case_id")
    .agg(
        [
            pl.len().alias("bureau_b2_payment_record_count"),

            pl.col("pmts_dpdvalue_108P")
            .max()
            .alias("bureau_b2_payment_dpd_max"),

            pl.col("pmts_dpdvalue_108P")
            .mean()
            .alias("bureau_b2_payment_dpd_mean"),

            pl.col("pmts_pmtsoverdue_635A")
            .max()
            .alias("bureau_b2_payment_overdue_max"),

            pl.col("pmts_pmtsoverdue_635A")
            .sum()
            .alias("bureau_b2_payment_overdue_sum"),

            pl.col("num_group1")
            .max()
            .alias("bureau_b2_num_group1_max"),

            pl.col("num_group2")
            .max()
            .alias("bureau_b2_num_group2_max"),
        ]
    )
)

print("Merging B1 and B2 features...")

features = b1_features.join(
    b2_features,
    on="case_id",
    how="outer",
    coalesce=True,
)

count_cols = [
    "bureau_b1_count",
    "bureau_b2_payment_record_count",
    "bureau_b2_num_group1_max",
    "bureau_b2_num_group2_max",
]

for col in count_cols:
    if col in features.columns:
        features = features.with_columns(pl.col(col).fill_null(0))

print("Final bureau B features:", features.shape)

features.write_parquet(OUT_PATH)

print(f"Saved -> {OUT_PATH}")