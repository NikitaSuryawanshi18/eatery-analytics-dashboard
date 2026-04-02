"""CLI: evaluate models, log MLflow run, and register model artifact."""

from __future__ import annotations

import argparse
import json

from milk_dashboard.data.file_source import FileDataSource
from milk_dashboard.pipeline import evaluate_fit_and_log, prepare_from_cleaned_file, prepare_from_source


def main() -> None:
    parser = argparse.ArgumentParser(description="Run model experiment and log to MLflow")
    parser.add_argument("--sales", required=False, help="Path to sales CSV/XLSX file")
    parser.add_argument("--ingredients", required=False, help="Path to ingredient XLSX/CSV file")
    parser.add_argument(
        "--input-csv",
        required=False,
        help="Path to pre-cleaned input file (CSV/Parquet/XLSX). If set, raw sales/ingredients are ignored.",
    )
    parser.add_argument("--model", default="prophet", help="Model key from registry")
    parser.add_argument("--tracking-uri", default=None, help="MLflow tracking URI")
    parser.add_argument("--experiment-name", default="milk_models", help="MLflow experiment name")
    parser.add_argument("--registered-model-name", default="milk_forecast", help="MLflow registered model name")
    parser.add_argument("--holdout-days", type=int, default=14)
    parser.add_argument("--min-train-days", type=int, default=30)
    parser.add_argument("--horizon-days", type=int, default=7)
    parser.add_argument("--step-days", type=int, default=7)
    parser.add_argument("--max-splits", type=int, default=5)
    parser.add_argument("--source-id", default="default_files")
    parser.add_argument("--fill-missing-days", action="store_true", help="Fill missing dates with zero gallons")
    parser.add_argument(
        "--apply-qty-multiplier",
        action="store_true",
        help="Multiply milk mapping by qty before daily aggregation",
    )
    args = parser.parse_args()

    if args.input_csv:
        prepared = prepare_from_cleaned_file(
            cleaned_file_path=args.input_csv,
            source_id=args.source_id,
        )
        source_config = {
            "source_id": args.source_id,
            "input_csv": args.input_csv,
            "source_mode": "cleaned_input",
        }
        preprocess_config = {"source_mode": "cleaned_input"}
    else:
        if not args.sales or not args.ingredients:
            parser.error("Either provide --input-csv OR both --sales and --ingredients.")

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
        source_config = {
            "source_id": args.source_id,
            "sales_path": args.sales,
            "ingredients_path": args.ingredients,
            "source_mode": "raw_plus_ingredients",
        }
        preprocess_config = {
            "fill_missing_days": args.fill_missing_days,
            "apply_qty_multiplier": args.apply_qty_multiplier,
            "source_mode": "raw_plus_ingredients",
        }

    split_config = {
        "holdout_days": args.holdout_days,
        "min_train_days": args.min_train_days,
        "horizon_days": args.horizon_days,
        "step_days": args.step_days,
        "max_splits": args.max_splits,
    }

    evaluation, run_info = evaluate_fit_and_log(
        daily_df=prepared.daily_df,
        model_name=args.model,
        source_config=source_config,
        preprocess_config=preprocess_config,
        split_config=split_config,
        experiment_name=args.experiment_name,
        registered_model_name=args.registered_model_name,
        tracking_uri=args.tracking_uri,
    )

    payload = {
        "run_info": {
            "run_id": run_info.run_id,
            "model_uri": run_info.model_uri,
            "registered_model_name": run_info.registered_model_name,
            "registered_model_version": run_info.registered_model_version,
        },
        "metrics": evaluation.metric_summary(),
    }
    print(json.dumps(payload, indent=2))


if __name__ == "__main__":
    main()
