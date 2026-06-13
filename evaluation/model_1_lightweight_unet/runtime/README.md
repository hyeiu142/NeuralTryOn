# Model 1 Runtime

These stages reuse the architectures and loaded checkpoints from
`notebooks/02_models/model_1_lightweight_unet.ipynb`.

Run the Model 1 notebook through the model-loading/training cells first. Then
execute these files in the same notebook namespace:

```text
00_install_dependencies.py
01_configure_paths.py
02_prepare_runtime.py
03_inference.py
```

The runtime validates:

```text
processor
model_unet
model_gmm
model_tom or TOM checkpoint
TOM_Generator
TPSGridGen
```

The main contract is:

```python
run_inference(person_id, cloth_id, mode="paired", return_debug=False)
run_inference(person_id, cloth_id, mode="unpaired", return_debug=False)
```

It returns an RGB NumPy image in `[0, 1]`.
