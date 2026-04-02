"""CLI: prepare cleaned daily dataset from source files."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from milk_dashboard.data.file_source import FileDataSource
from milk_dashboard.pipeline import prepare_from_source


def main() -> None:
    parser = argparse.ArgumentParser(description="Prepare cleaned daily milk dataset")
    parser.add_argument("--sales", required=True, help="Path to sales CSV/XLSX file")
    parser.add_argument("--ingredients", required=True, help="Path to ingredient XLSX/CSV file")
    parser.add_argument("--out-dir", default="artifacts/data", help="Output directory")
    parser.add_argument("--source-id", default="default_files", help="Logical source id for metadata")
    parser.add_argument("--fill-missing-days", action="store_true", help="Fill missing dates with zero gallons")
    parser.add_argument(
        "--apply-qty-multiplier",
        action="store_true",
        help="Multiply milk mapping by qty before daily aggregation",
    )
    args = parser.parse_args()

    source = FileDataSource(
        sales_path=args.sales,
        ingredients_path=args.ingredients,
        _source_id=args.source_id,
    )
    prepared = prepare_from_source(
        source=source,
        fill_missing_days=args.fill_missing_days,
        apply_qty_multiplier=args.apply_qty_multiplier,
    )

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    daily_path = out_dir / "daily.parquet"
    schema_path = out_dir / "schema.json"

    prepared.daily_df.to_parquet(daily_path, index=False)
    schema_path.write_text(json.dumps(prepared.schema, indent=2), encoding="utf-8")

    print(f"Wrote: {daily_path}")
    print(f"Wrote: {schema_path}")


if __name__ == "__main__":
    main()
