"""Unified command line interface for production-oriented VTO operations."""

import argparse
import json
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from src.artifacts import package_run
from src.config import load_experiment_config
from src.models import MODEL_SPECS
from src.runtime import inspect_runtime
from src.tracking import RunTracker


def main() -> None:
    """Execute a project operation."""
    parser = argparse.ArgumentParser(prog="vto")
    commands = parser.add_subparsers(dest="command", required=True)

    for command in ("validate", "preflight", "init-run"):
        child = commands.add_parser(command)
        child.add_argument("--config", required=True, type=Path)

    package = commands.add_parser("package-run")
    package.add_argument("--run-dir", required=True, type=Path)
    commands.add_parser("models")

    args = parser.parse_args()
    if args.command == "models":
        for spec in MODEL_SPECS.values():
            print(f"{spec.key}: {spec.display_name} [{spec.production_adapter_status}]")
        return
    if args.command == "package-run":
        print(package_run(args.run_dir))
        return

    config = load_experiment_config(args.config)
    if args.command == "validate":
        print(f"Valid configuration: {config['name']}")
    elif args.command == "preflight":
        print(json.dumps(inspect_runtime(config, PROJECT_ROOT), indent=2))
    elif args.command == "init-run":
        print(RunTracker(config, PROJECT_ROOT).start())


if __name__ == "__main__":
    main()

