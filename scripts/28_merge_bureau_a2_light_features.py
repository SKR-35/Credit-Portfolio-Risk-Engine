"""
Merge Bureau A2 light features into master training dataset.

v0.1 note:
----------
Only lightweight Bureau A2 aggregates are merged because the full
feature set exceeds the memory limits of the local development machine.

The goal is to measure incremental signal contribution from Bureau A2.
"""

from pathlib import Path
import polars as pl

BASE_PATH = (
    "data/processed/"
    "train_static_person_applprev_bureaub_a1_light_features.parquet"
)

A2_PATH = (
    "data/processed/"
    "bureau_a2_features_light.parquet"
)

OUT_PATH = (
    "data/processed/"
    "train_static_person_applprev_bureaub_a1_a2_light_features.parquet"
)

print("Loading datasets...")

base = pl.read_parquet(BASE_PATH)
a2 = pl.read_parquet(A2_PATH)

print("Base:", base.shape)
print("Bureau A2 light:", a2.shape)

print("Joining...")

merged = base.join(
    a2,
    on="case_id",
    how="left"
)

print("Joined:", merged.shape)

new_cols = [
    c
    for c in a2.columns
    if c != "case_id"
]

print("Filling missing Bureau A2 values...")

merged = merged.with_columns(
    [
        pl.col(col)
        .fill_null(0)
        .alias(col)
        for col in new_cols
    ]
)

merged.write_parquet(
    OUT_PATH
)

print(f"Saved -> {OUT_PATH}")