# Production-Oriented Architecture

## Purpose

The completed Kaggle notebooks remain immutable experiment records. A separate
production-oriented layer now standardizes configuration, data contracts,
training lifecycle, evaluation, inference, tracking, and artifact transfer.
This avoids changing successful notebook workflows while providing a clean
migration path toward script-based execution.

## System Layers

```text
configs/experiments/*.yaml
        |
        v
src.config -> resolved reproducible configuration
        |
        +---------------------+----------------------+------------------+
        v                     v                      v                  v
src.data                 src.models             src.training      src.evaluation
manifests + paths        contract + registry    reusable engine   paired metrics
        |                     |                      |                  |
        +---------------------+----------------------+------------------+
                                      |
                                      v
                                src.tracking
                       config snapshot + metrics + artifacts
```

## Configuration Strategy

Configuration follows separation of concerns:

- `configs/data/`: manifests, split policies, image size, and workers.
- `configs/models/`: architecture and model identity.
- `configs/experiments/`: training, inference, evaluation, and ablation values.
- `configs/tracking/`: run directory and registry policy.

Each experiment file references the shared components. At runtime,
`load_experiment_config` resolves them into one immutable `config.yaml`
snapshot stored with the run.

## Run Lifecycle

Every remote or local execution receives one run directory:

```text
experiments/runs/<timestamp>_<experiment>/
├── config.yaml
├── metadata.json
├── metrics.jsonl
├── summary.json
├── checkpoints/
├── logs/
└── artifacts/
```

`metrics.jsonl` is append-only, which makes interrupted runs easier to inspect.
`registry.csv` stores the compact run index. Large run directories are excluded
from Git and can be published as Kaggle Datasets.

## Kaggle GPU Workflow

```bash
git clone <repository-url> /kaggle/working/NeuralTryOn
cd /kaggle/working/NeuralTryOn
python scripts/vto.py preflight --config configs/experiments/model_3_default.yaml
python scripts/vto.py init-run --config configs/experiments/model_3_default.yaml
```

The existing notebooks remain the authoritative training implementation.
During migration, notebook metrics and checkpoints can be written into the run
directory. After training, package the run:

```bash
python scripts/vto.py package-run --run-dir experiments/runs/<run_id>
```

Publish the resulting archive or checkpoint directory as a Kaggle Dataset.
W&B may remain enabled for remote dashboards; the local run structure keeps an
independent, portable experiment record.

## Current Migration Status

| Layer | Status |
| --- | --- |
| Hierarchical configuration | Implemented |
| Dataset manifest validation and pairing policies | Implemented |
| Kaggle/local path discovery | Implemented |
| Run tracking and registry | Implemented |
| Generic PyTorch training engine | Implemented |
| Generic paired evaluator and inference runner | Implemented |
| Model registry and contracts | Implemented |
| Model 3 train, inference, and paired-evaluation adapters | Implemented; Kaggle GPU smoke test pending |
| Model 1 and Model 2 training adapters | Retained in immutable notebooks |
| Model-specific evaluation workflows | Implemented under `evaluation/` |

The CLI intentionally refuses to launch a different model implementation until
its adapter has been migrated from the corresponding notebook. This prevents
silent differences between reported experiments and script-based reruns.

Adapter acceptance criteria are documented in
[model_adapter_migration.md](model_adapter_migration.md).

Model 3 production training commands are documented in
[model_3_production_training.md](model_3_production_training.md).
