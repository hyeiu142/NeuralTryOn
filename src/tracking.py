"""Local, run-oriented experiment tracking utilities."""

from __future__ import annotations

import csv
import json
import platform
import subprocess
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import yaml


REGISTRY_FIELDS = (
    "run_id",
    "experiment",
    "status",
    "started_at",
    "finished_at",
    "git_commit",
    "run_dir",
)


class RunTracker:
    """Create a run directory and record metrics, metadata, and summaries."""

    def __init__(self, config: dict[str, Any], project_root: str | Path = ".") -> None:
        self.config = config
        self.project_root = Path(project_root).resolve()
        self.started_at = datetime.now(timezone.utc)
        self.run_id = f"{self.started_at:%Y%m%dT%H%M%S%fZ}_{_slug(config['name'])}"
        run_root = self.project_root / config["tracking"]["run_root"]
        self.run_dir = run_root / self.run_id
        self.registry_path = self.project_root / config["tracking"]["registry"]

    def start(self) -> Path:
        """Initialize directories and immutable run metadata."""
        for name in ("checkpoints", "logs", "artifacts"):
            (self.run_dir / name).mkdir(parents=True, exist_ok=True)

        _write_yaml(self.run_dir / "config.yaml", self.config)
        _write_json(
            self.run_dir / "metadata.json",
            {
                "run_id": self.run_id,
                "experiment": self.config["name"],
                "status": "running",
                "started_at": self.started_at.isoformat(),
                "git_commit": _git_commit(self.project_root),
                "python": platform.python_version(),
                "platform": platform.platform(),
            },
        )
        (self.run_dir / "metrics.jsonl").touch()
        self._register(status="running")
        return self.run_dir

    def log_metrics(self, metrics: dict[str, float], step: int, split: str) -> None:
        """Append one metric event without rewriting previous history."""
        event = {
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "step": step,
            "split": split,
            "metrics": metrics,
        }
        with (self.run_dir / "metrics.jsonl").open("a", encoding="utf-8") as file:
            file.write(json.dumps(event, sort_keys=True) + "\n")

    def finish(self, summary: dict[str, Any], status: str = "completed") -> None:
        """Save the final summary and update run metadata and registry."""
        finished_at = datetime.now(timezone.utc)
        _write_json(self.run_dir / "summary.json", summary)
        metadata = json.loads((self.run_dir / "metadata.json").read_text())
        metadata.update({"status": status, "finished_at": finished_at.isoformat()})
        _write_json(self.run_dir / "metadata.json", metadata)
        self._register(status=status, finished_at=finished_at.isoformat())

    def _register(self, status: str, finished_at: str = "") -> None:
        self.registry_path.parent.mkdir(parents=True, exist_ok=True)
        rows = _read_registry(self.registry_path)
        row = {
            "run_id": self.run_id,
            "experiment": self.config["name"],
            "status": status,
            "started_at": self.started_at.isoformat(),
            "finished_at": finished_at,
            "git_commit": _git_commit(self.project_root),
            "run_dir": str(self.run_dir.relative_to(self.project_root)),
        }
        rows[self.run_id] = row
        with self.registry_path.open("w", newline="", encoding="utf-8") as file:
            writer = csv.DictWriter(file, fieldnames=REGISTRY_FIELDS)
            writer.writeheader()
            writer.writerows(rows.values())


def _read_registry(path: Path) -> dict[str, dict[str, str]]:
    if not path.exists():
        return {}
    with path.open(newline="", encoding="utf-8") as file:
        return {row["run_id"]: row for row in csv.DictReader(file) if row.get("run_id")}


def _git_commit(project_root: Path) -> str:
    result = subprocess.run(
        ["git", "rev-parse", "--short", "HEAD"],
        cwd=project_root,
        capture_output=True,
        text=True,
        check=False,
    )
    return result.stdout.strip() if result.returncode == 0 else "unknown"


def _slug(value: str) -> str:
    return "".join(char if char.isalnum() else "_" for char in value.lower()).strip("_")


def _write_json(path: Path, value: dict[str, Any]) -> None:
    path.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def _write_yaml(path: Path, value: dict[str, Any]) -> None:
    path.write_text(yaml.safe_dump(value, sort_keys=False), encoding="utf-8")
