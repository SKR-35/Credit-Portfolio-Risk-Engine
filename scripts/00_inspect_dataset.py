from pathlib import Path
import polars as pl

DATA_DIR = Path("data/raw/parquet_files/train")

for file in sorted(DATA_DIR.glob("*.parquet")):
    try:
        df = pl.scan_parquet(file)

        print("=" * 80)
        print(file.name)

        schema = df.collect_schema()

        print(f"Columns: {len(schema)}")

        for col, dtype in schema.items():
            print(f"{col}: {dtype}")

    except Exception as e:
        print(file.name, e)