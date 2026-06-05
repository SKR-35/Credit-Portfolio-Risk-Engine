from pathlib import Path
import polars as pl

PERSON1 = Path(
    "data/raw/parquet_files/train/train_person_1.parquet"
)

PERSON2 = Path(
    "data/raw/parquet_files/train/train_person_2.parquet"
)

OUT_PATH = Path(
    "data/processed/person_features.parquet"
)

OUT_PATH.parent.mkdir(
    parents=True,
    exist_ok=True
)

print("Loading person tables...")

p1 = pl.read_parquet(PERSON1)
p2 = pl.read_parquet(PERSON2)

print("Creating aggregates...")

p1_features = (
    p1.group_by("case_id")
      .agg(
          [
              pl.count().alias("person1_record_count"),

              pl.col("mainoccupationinc_384A")
                .mean()
                .alias("person1_income_mean"),

              pl.col("mainoccupationinc_384A")
                .max()
                .alias("person1_income_max"),

              pl.col("mainoccupationinc_384A")
                .min()
                .alias("person1_income_min"),

              pl.col("childnum_185L")
                .max()
                .alias("person1_childnum_max"),
          ]
      )
)

p2_features = (
    p2.group_by("case_id")
      .agg(
          [
              pl.count().alias("person2_record_count"),

              pl.col("num_group1")
                .max()
                .alias("person2_num_group1_max"),

              pl.col("num_group2")
                .max()
                .alias("person2_num_group2_max"),
          ]
      )
)

person_features = (
    p1_features.join(
        p2_features,
        on="case_id",
        how="left"
    )
)

print(person_features.shape)

person_features.write_parquet(
    OUT_PATH
)

print(f"Saved -> {OUT_PATH}")