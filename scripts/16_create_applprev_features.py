from pathlib import Path
import polars as pl

APPL1 = Path(
    "data/raw/parquet_files/train/train_applprev_1_0.parquet"
)

APPL2 = Path(
    "data/raw/parquet_files/train/train_applprev_1_1.parquet"
)

OUT_PATH = Path(
    "data/processed/applprev_features.parquet"
)

print("Loading applprev tables...")

a1 = pl.read_parquet(APPL1)
a2 = pl.read_parquet(APPL2)

applprev = pl.concat(
    [a1, a2],
    how="vertical"
)

print(applprev.shape)

print("Creating aggregates...")

features = (
    applprev
    .group_by("case_id")
    .agg(
        [
            pl.len().alias("applprev_count"),

            pl.col("credamount_590A")
            .mean()
            .alias("applprev_credamount_mean"),

            pl.col("credamount_590A")
            .max()
            .alias("applprev_credamount_max"),

            pl.col("annuity_853A")
            .mean()
            .alias("applprev_annuity_mean"),

            pl.col("annuity_853A")
            .max()
            .alias("applprev_annuity_max"),

            pl.col("currdebt_94A")
            .mean()
            .alias("applprev_currdebt_mean"),

            pl.col("outstandingdebt_522A")
            .mean()
            .alias("applprev_outstanding_mean"),

            pl.col("actualdpd_943P")
            .max()
            .alias("applprev_max_dpd"),

            pl.col("tenor_203L")
            .mean()
            .alias("applprev_tenor_mean"),

            pl.col("pmtnum_8L")
            .max()
            .alias("applprev_pmtnum_max"),
        ]
    )
)

print(features.shape)

OUT_PATH.parent.mkdir(
    parents=True,
    exist_ok=True
)

features.write_parquet(
    OUT_PATH
)

print(f"Saved -> {OUT_PATH}")