# Checkpoints and External Artifacts

Large checkpoints are excluded from Git and should be attached as Kaggle
Datasets before running inference.

## Model 3: SD + LoRA

Final evaluated checkpoint:

```text
/kaggle/input/datasets/khoaanh1234/ckpt-epoch-12-yen
```

Expected checkpoint contents:

```text
checkpoint_latest/
├── unet_lora/
├── conv_in.pt
├── perceiver.pt
└── cloth_spatial.pt
```

Paired test result dataset:

```text
khoaanh1234/vto-sd-lora-epoch12-paired-three-column-gallery
```

Document the final Kaggle Dataset links for Models 1 and 2 here once published.

## Model 2: Pix2Pix

Evaluated checkpoint files are loaded from:

```text
/kaggle/input/datasets/vnhttin/module-trained/
├── best_gmm.pth
├── best_seg_stage1.pth
└── best_tom_stage2.pth
```

The notebook also resumed Stage 2 training from epoch-17 and epoch-28
checkpoint datasets before completing epoch 30.
