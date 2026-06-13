# Experiment Configurations

This directory records the hyperparameters used by the three model notebooks.
The YAML files are reference manifests for review, comparison, and future
refactoring.

The current notebooks remain self-contained and do not load these files at
runtime. Update a YAML file whenever the corresponding notebook configuration
changes.

| Configuration | Source notebook |
| --- | --- |
| `model_1_lightweight_unet.yaml` | `notebooks/02_models/model_1_lightweight_unet.ipynb` |
| `model_2_pix2pix.yaml` | `notebooks/02_models/model_2_pix2pix.ipynb` |
| `model_3_sd_lora.yaml` | `notebooks/02_models/model_3_sd_lora.ipynb` |

