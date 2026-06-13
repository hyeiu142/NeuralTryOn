# Evaluation Pipelines

Each model owns its inference adapter and evaluation workflow in this directory.
Shared metric definitions and visualization helpers live under `src/`.

Currently implemented:

```text
model_1_lightweight_unet/
model_2_pix2pix/
model_3_sd_lora/
common_comparison/
```

Models 1 and 2 contain paired and unpaired evaluation workflows for the
lightweight CNN and Pix2Pix pipelines. Model 3 contains the Stable Diffusion +
LoRA workflow.

`common_comparison/` generates a clearly labeled overview from the currently
reported results.
