# Runtime

These stages construct the shared inference runtime used by both evaluation
workflows.

```text
00_install_dependencies.py  Dependency check for a fresh Kaggle session
01_configure_paths.py       Dataset/checkpoint discovery and validation
02_load_models.py           Model reconstruction and checkpoint loading
03_inference.py             Preprocessing and run_inference definition
```

After all runtime stages execute, the shared notebook namespace contains:

```text
run_inference
vae
tokenizer
text_encoder
unet
image_encoder
perceiver
cloth_spatial_proj
scheduler
VITON_ROOT
CSV_ROOT
CAPTION_ROOT
OUTPUT_DIR
```

`run_inference(person_id, cloth_id, split="test", return_debug=False)` is the
main inference contract. It returns an RGB NumPy array in `[0, 1]`.
