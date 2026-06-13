# Model 2 Runtime

These stages reuse the architecture classes defined by
`notebooks/02_models/model_2_pix2pix.ipynb` and load the three evaluated
checkpoints:

```text
best_gmm.pth
best_seg_stage1.pth
best_tom_stage2.pth
```

Run after the Model 2 architecture-definition cells:

```text
00_install_dependencies.py
01_configure_paths.py
02_prepare_runtime.py
03_inference.py
```

The main inference contract is:

```python
run_inference(person_id, cloth_id, return_debug=False)
```

It runs GMM cloth warping, Stage 1 shape generation, and Stage 2 Pix2Pix
generation, then returns an RGB NumPy image in `[0, 1]`.
