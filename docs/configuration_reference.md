# Configuration Reference

The `configs/` directory records the settings that affect architecture,
training, evaluation, reproducibility, and artifact recovery. The three source
notebooks remain unchanged and self-contained.

## Configuration Layers

| Layer | Responsibility |
| --- | --- |
| `configs/data/viton_hd.yaml` | Shared VITON-HD paths, modalities, labels, normalization, and pose settings |
| `configs/models/*.yaml` | Architecture-only settings for each model |
| `configs/experiments/*.yaml` | Data split, augmentation, loaders, training stages, losses, checkpoints, inference, and evaluation |
| `configs/tracking/default.yaml` | Shared local run-recording defaults |

Each experiment configuration references one data, model, and tracking
configuration. `src.config.load_experiment_config` resolves these references
into a single snapshot that can be stored with a future run.

## Notebook Mapping

| Experiment configuration | Source notebook |
| --- | --- |
| `configs/experiments/model_1_default.yaml` | `notebooks/02_models/model_1_lightweight_unet.ipynb` |
| `configs/experiments/model_2_default.yaml` | `notebooks/02_models/model_2_pix2pix.ipynb` |
| `configs/experiments/model_3_default.yaml` | `notebooks/02_models/model_3_sd_lora.ipynb` |

## Scope

The YAML files capture values required to explain or reproduce an experiment:
network dimensions, input composition, data splits, augmentation, optimizer
settings, losses, schedulers, stopping criteria, checkpoint names, inference,
and evaluation protocols.

Notebook display constants, temporary paths, plotting styles, debugging flags,
and one-off interactive cell state are intentionally excluded. Completed-run
artifacts and checkpoints remain external when they are too large for Git.

## Using a Resolved Configuration

```python
from src.config import load_experiment_config

config = load_experiment_config("configs/experiments/model_3_default.yaml")
```

The returned dictionary contains `data`, `model`, and `tracking` sections in
addition to the experiment-specific fields. Future script-based runs should
store this resolved configuration beside their logs and checkpoints.
