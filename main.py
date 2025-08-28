import argparse
from pathlib import Path

from cowculator.pipeline import (
    fetch_data_cli,
    predict_next_cli,
    train_cli,
    export_edge_cli,
)


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="CaterCow order cost modeling")
    sub = p.add_subparsers(dest="cmd", required=True)

    fetch = sub.add_parser("fetch", help="Download raw API data")
    fetch.add_argument("--limit", type=int, default=100, help="API page size")
    fetch.add_argument(
        "--max-pages", type=int, default=100, help="Safety cap for pages"
    )
    fetch.add_argument(
        "--outdir",
        type=Path,
        default=Path("data/raw"),
        help="Directory to store raw JSON",
    )

    train = sub.add_parser("train", help="Prepare data and train model")
    train.add_argument(
        "--raw",
        type=Path,
        default=Path("data/raw"),
        help="Directory with raw JSON files",
    )
    train.add_argument(
        "--processed",
        type=Path,
        default=Path("data/processed/orders.parquet"),
        help="Path to save processed parquet",
    )
    train.add_argument(
        "--artifacts",
        type=Path,
        default=Path("artifacts"),
        help="Directory to save model + metrics",
    )

    predict = sub.add_parser("predict-next", help="Predict tomorrow's cost")
    predict.add_argument(
        "--artifacts",
        type=Path,
        default=Path("artifacts"),
        help="Directory with model + weekday stats",
    )

    export = sub.add_parser(
        "export-edge", help="Export compact JSON bundle for browser inference"
    )
    export.add_argument(
        "--artifacts",
        type=Path,
        default=Path("artifacts"),
        help="Directory with trained artifacts",
    )
    export.add_argument(
        "--out",
        type=Path,
        default=Path("web/edge_bundle.json"),
        help="Output path for bundle JSON (served by Pages)",
    )

    return p


def main():
    parser = build_parser()
    args = parser.parse_args()
    if args.cmd == "fetch":
        fetch_data_cli(args)
    elif args.cmd == "train":
        train_cli(args)
    elif args.cmd == "predict-next":
        predict_next_cli(args)
    elif args.cmd == "export-edge":
        export_edge_cli(args)
    else:
        parser.error("Unknown command")


if __name__ == "__main__":
    main()
