"""Run tracking lifecycle tests."""

import json

from src.tracking import RunTracker


def test_run_tracker_lifecycle(tmp_path) -> None:
    config = {
        "name": "test_run",
        "tracking": {
            "run_root": "experiments/runs",
            "registry": "experiments/registry.csv",
        },
    }
    tracker = RunTracker(config, tmp_path)
    run_dir = tracker.start()
    tracker.log_metrics({"loss": 1.0}, step=1, split="train")
    tracker.finish({"best_loss": 1.0})

    metadata = json.loads((run_dir / "metadata.json").read_text())
    assert metadata["status"] == "completed"
    assert (run_dir / "config.yaml").exists()
    assert (run_dir / "summary.json").exists()

