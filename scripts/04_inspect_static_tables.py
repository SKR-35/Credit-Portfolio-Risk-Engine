from pathlib import Path
import polars as pl

STATIC_FILES = [
    Path("data/raw/parquet_files/train/train_static_0_0.parquet"),
    Path("data/raw/parquet_files/train/train_static_0_1.parquet"),
]

REPORT_DIR = Path("reports/inspection")
REPORT_DIR.mkdir(parents=True, exist_ok=True)


def write_and_print(text: str, file_handle):
    print(text)
    file_handle.write(text + "\n")


for file in STATIC_FILES:

    report_file = REPORT_DIR / f"{file.stem}_report.txt"

    with open(report_file, "w", encoding="utf-8") as f:

        df = pl.read_parquet(file)

        write_and_print("=" * 100, f)
        write_and_print(file.name, f)

        write_and_print("\nSHAPE", f)
        write_and_print(str(df.shape), f)

        write_and_print("\nCASE_ID CHECK", f)
        write_and_print(
            f"unique case_id: {df['case_id'].n_unique()}",
            f
        )
        write_and_print(
            f"rows: {df.height}",
            f
        )

        write_and_print("\nDTYPES", f)
        write_and_print(str(df.schema), f)

        write_and_print("\nFIRST 5 ROWS", f)
        write_and_print(str(df.head()), f)

        write_and_print("\nMISSING VALUES TOP 30", f)

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
            str(missing.head(30)),
            f
        )

    print(f"\nSaved report -> {report_file}\n")