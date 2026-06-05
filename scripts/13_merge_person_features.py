from pathlib import Path
import polars as pl

STATIC_PATH = Path(
    "data/processed/train_static_features.parquet"
)

PERSON_PATH = Path(
    "data/processed/person_features.parquet"
)

OUT_PATH = Path(
    "data/processed/train_static_person_features.parquet"
)

print("Loading datasets...")

static = pl.read_parquet(STATIC_PATH)
person = pl.read_parquet(PERSON_PATH)

print("Static:", static.shape)
print("Person:", person.shape)

print("Joining...")

df = static.join(
    person,
    on="case_id",
    how="left"
)

print("Joined:", df.shape)

print("Filling person count missing values...")

count_cols = [
    "person1_record_count",
    "person2_record_count",
    "person2_num_group1_max",
    "person2_num_group2_max",
]

for col in count_cols:
    if col in df.columns:
        df = df.with_columns(
            pl.col(col).fill_null(0)
        )

OUT_PATH.parent.mkdir(
    parents=True,
    exist_ok=True
)

df.write_parquet(OUT_PATH)

print(f"Saved -> {OUT_PATH}")