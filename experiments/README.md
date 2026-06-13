# Experiment Tracking

This directory is the central index for model configurations and training logs.
It complements the self-contained notebooks without changing their runtime
behavior.

## Central Files

- `hyperparameter_summary.csv`: compact cross-model configuration table.
- `../configs/`: detailed reference manifests copied from the model notebooks.
- `../docs/experiments.md`: completed-run status, results, and observations.
- `../results/convergence/`: report-ready training and validation curves.
- `../results/metrics/`: report-ready evaluation summaries.

## Log Collection

Raw logs are intentionally excluded from Git because TensorBoard and W&B files
can become large. After a run, place exported logs under:

```text
experiments/logs/
├── model_1_lightweight_unet/
├── model_2_pix2pix/
└── model_3_sd_lora/
```

Model 1 writes TensorBoard events using
`runs/TOM_Experiment_{config_name}`. Model 2 currently records history in the
notebook and checkpoints. Model 3 sends training metrics to W&B project
`VTO-Model4-SD-LoRA`.

## Review Checklist

1. Compare planned settings in `configs/` with the notebook run.
2. Confirm the final checkpoint and completed epoch.
3. Inspect training and validation curves in `results/convergence/`.
4. Inspect quantitative summaries in `results/metrics/`.
5. Record conclusions and run status in `docs/experiments.md`.

