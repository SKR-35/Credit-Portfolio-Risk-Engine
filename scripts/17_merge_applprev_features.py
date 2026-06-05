from pathlib import Path
import polars as pl

BASE_PATH = Path(
    "data/processed/train_static_person_features.parquet"
)

APPLPREV_PATH = Path(
    "data/processed/applprev_features.parquet"
)

OUT_PATH = Path(
    "data/processed/train_static_person_applprev_features.parquet"
)

print("Loading datasets...")

base = pl.read_parquet(BASE_PATH)
applprev = pl.read_parquet(APPLPREV_PATH)

print("Base:", base.shape)
print("ApplPrev:", applprev.shape)

print("Joining...")

df = base.join(
    applprev,
    on="case_id",
    how="left"
)

print("Joined:", df.shape)

numeric_fill = [
    "applprev_count",
    "applprev_credamount_mean",
    "applprev_credamount_max",
    "applprev_annuity_mean",
    "applprev_annuity_max",
    "applprev_currdebt_mean",
    "applprev_outstanding_mean",
    "applprev_max_dpd",
    "applprev_tenor_mean",
    "applprev_pmtnum_max",
]

for col in numeric_fill:
    if col in df.columns:
        df = df.with_columns(
            pl.col(col).fill_null(0)
        )

df.write_parquet(OUT_PATH)

print(f"Saved -> {OUT_PATH}")