from pathlib import Path
import polars as pl

FILES = [
    Path("data/raw/parquet_files/train/train_applprev_1_0.parquet"),
    Path("data/raw/parquet_files/train/train_applprev_1_1.parquet"),
    Path("data/raw/parquet_files/train/train_applprev_2.parquet"),
]

REPORT_DIR = Path("reports/inspection")
REPORT_DIR.mkdir(parents=True, exist_ok=True)


def write(text, f):
    print(text)
    f.write(str(text) + "\n")


for file in FILES:
    report_file = REPORT_DIR / f"{file.stem}_report.txt"

    with open(report_file, "w", encoding="utf-8") as f:
        df = pl.read_parquet(file)

        write("=" * 100, f)
        write(file.name, f)

        write("\nSHAPE", f)
        write(df.shape, f)

        write("\nCASE_ID CHECK", f)
        write(f"rows: {df.height}", f)
        write(f"unique case_id: {df['case_id'].n_unique()}", f)

        for group_col in ["num_group1", "num_group2"]:
            if group_col in df.columns:
                write(f"unique {group_col}: {df[group_col].n_unique()}", f)

        write("\nDTYPES", f)
        write(df.schema, f)

        write("\nFIRST 5 ROWS", f)
        write(df.head(), f)

        write("\nMISSING VALUES TOP 30", f)

        missing = (
            df.null_count()
            .transpose(include_header=True)
            .rename({"column": "feature", "column_0": "null_count"})
            .with_columns((pl.col("null_count") / df.height).alias("null_ratio"))
            .sort("null_count", descending=True)
        )

        write(missing.head(30), f)

    print(f"Saved -> {report_file}")