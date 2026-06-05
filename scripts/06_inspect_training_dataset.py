from pathlib import Path
import polars as pl

DATA_PATH = Path(
    "data/processed/train_static_features.parquet"
)

REPORT_DIR = Path("reports/inspection")
REPORT_DIR.mkdir(parents=True, exist_ok=True)

REPORT_FILE = REPORT_DIR / "training_dataset_report.txt"


def write_and_print(text, f):
    print(text)
    f.write(str(text) + "\n")


with open(REPORT_FILE, "w", encoding="utf-8") as f:

    df = pl.read_parquet(DATA_PATH)

    write_and_print("=" * 100, f)
    write_and_print("TRAINING DATASET REPORT", f)

    write_and_print("\nSHAPE", f)
    write_and_print(df.shape, f)

    numeric_cols = []
    categorical_cols = []
    boolean_cols = []

    for col, dtype in df.schema.items():

        dtype_str = str(dtype)

        if dtype_str.startswith(("Int", "Float")):
            numeric_cols.append(col)

        elif dtype_str == "Boolean":
            boolean_cols.append(col)

        else:
            categorical_cols.append(col)

    write_and_print("\nCOLUMN SUMMARY", f)
    write_and_print(f"Numeric: {len(numeric_cols)}", f)
    write_and_print(f"Categorical: {len(categorical_cols)}", f)
    write_and_print(f"Boolean: {len(boolean_cols)}", f)

    write_and_print("\nTARGET DISTRIBUTION", f)

    target_dist = df["target"].value_counts()

    write_and_print(target_dist, f)

    write_and_print("\nTOP 30 MISSING FEATURES", f)

    missing = (
        df.null_count()
        .transpose(include_header=True)
        .rename(
            {
                "column": "feature",
                "column_0": "null_count"
            }
        )
        .with_columns(
            (
                pl.col("null_count") / df.height
            ).alias("null_ratio")
        )
        .sort(
            "null_count",
            descending=True
        )
    )

    write_and_print(
        missing.head(30),
        f
    )

    write_and_print("\nFIRST 10 FEATURES", f)

    write_and_print(
        df.columns[:10],
        f
    )

print(f"\nSaved report -> {REPORT_FILE}")