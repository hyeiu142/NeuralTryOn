# Experiment Tracking

Each training execution is represented by one immutable run directory:

```text
experiments/runs/<timestamp>_<experiment_name>/
├── config.yaml       Resolved configuration snapshot
├── metadata.json     Run identity, status, Git commit, and environment
├── metrics.jsonl     Append-only step and epoch metrics
├── summary.json      Final metrics and conclusions
├── checkpoints/      Model states
├── logs/             TensorBoard or text logs
└── artifacts/        Samples and run-specific figures
```

`registry.csv` is the compact, Git-friendly index of all runs. Raw run
directories are excluded from Git because they can contain large checkpoints
and logs. Final report-ready outputs belong in `results/`.

## Commands

Validate a configuration without starting training:

```bash
python scripts/manage_experiment.py validate configs/experiments/model_3_default.yaml
```

Create an empty tracked run:

```bash
python scripts/manage_experiment.py init configs/experiments/model_3_default.yaml
```

List registered runs:

```bash
python scripts/manage_experiment.py list
```

The current notebooks are intentionally unchanged. Future training scripts can
use `src.config` and `src.tracking` directly.

