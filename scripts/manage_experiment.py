"""Validate, initialize, and list tracked experiment runs."""

import argparse
import csv
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from src.config import load_experiment_config
from src.tracking import RunTracker


def main() -> None:
    """Run the experiment-management command line interface."""
    parser = argparse.ArgumentParser()
    subparsers = parser.add_subparsers(dest="command", required=True)

    for command in ("validate", "init"):
        child = subparsers.add_parser(command)
        child.add_argument("config", type=Path)
    subparsers.add_parser("list")

    args = parser.parse_args()
    if args.command == "validate":
        config = load_experiment_config(args.config)
        print(f"Valid configuration: {config['name']}")
    elif args.command == "init":
        config = load_experiment_config(args.config)
        print(RunTracker(config, PROJECT_ROOT).start())
    else:
        list_runs(PROJECT_ROOT / "experiments/registry.csv")


def list_runs(registry_path: Path) -> None:
    """Print the compact run registry."""
    with registry_path.open(newline="", encoding="utf-8") as file:
        rows = list(csv.DictReader(file))
    if not rows:
        print("No registered runs.")
        return
    for row in rows:
        print(f"{row['run_id']}  {row['status']}  {row['experiment']}")


if __name__ == "__main__":
    main()
