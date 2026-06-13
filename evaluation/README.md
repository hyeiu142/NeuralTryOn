# Evaluation Pipelines

Each model owns its inference adapter and evaluation workflow in this directory.
Shared metric definitions and visualization helpers live under `src/`.

Currently implemented:

```text
model_1_lightweight_unet/
model_3_sd_lora/
common_comparison/
```

Model 1 contains the Lightweight U-Net + GMM + TOM paired and unpaired
evaluation workflow. Model 3 contains the Stable Diffusion + LoRA workflow.

The common three-model comparison can be added after Model 2 exports
predictions using an agreed paired-test manifest.

`common_comparison/` currently generates a clearly labeled overview from the
completed Model 1 and Model 3 reported results.
