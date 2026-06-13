# Kaggle Operations

## Before Training

1. Enable a GPU accelerator.
2. Attach VITON-HD, cleaned CSV manifests, captions, and required checkpoints.
3. Clone the repository into `/kaggle/working`.
4. Run production preflight.
5. Record the Git commit and selected experiment configuration.

```bash
python scripts/vto.py preflight --config configs/experiments/model_3_default.yaml
python scripts/train.py --config configs/experiments/model_3_default.yaml --dry-run
```

## During Training

- Use the immutable model notebook as the authoritative implementation.
- Log step and epoch metrics to W&B or TensorBoard.
- Save resumable checkpoints and the best validation checkpoint.
- Preserve the resolved config, random seed, and completed epoch.

The reusable `RunTracker` API is available when a notebook is intentionally
connected to the production layer:

```python
from src.config import load_experiment_config
from src.tracking import RunTracker

config = load_experiment_config("configs/experiments/model_3_default.yaml")
tracker = RunTracker(config)
run_dir = tracker.start()
tracker.log_metrics({"train_loss": 0.2, "validation_loss": 0.3}, step=1, split="epoch")
tracker.finish({"best_epoch": 1, "best_validation_loss": 0.3})
```

## After Training

1. Run paired evaluation and unpaired holdout inference.
2. Export compact metrics and report-ready figures.
3. Package the complete run.
4. Publish large checkpoints and galleries as Kaggle Datasets.
5. Commit only code, configs, registry updates, and compact results to Git.

