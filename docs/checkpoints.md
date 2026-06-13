# Checkpoints and External Artifacts

Large checkpoints are excluded from Git and should be attached as Kaggle
Datasets before running inference.

## Model 3: SD + LoRA

Best checkpoint:

- Kaggle Dataset: [Model 3 Best Checkpoint](https://www.kaggle.com/datasets/hypivepiu/model-3-bestbest?select=checkpoint_latest)
- Kaggle input path: `/kaggle/input/datasets/hypivepiu/model-3-bestbest`
- Selected directory: `checkpoint_latest`

Expected checkpoint contents:

```text
checkpoint_latest/
├── unet_lora/
├── conv_in.pt
├── perceiver.pt
└── cloth_spatial.pt
```

Paired test result dataset:

- Kaggle Dataset: [Epoch 12 Paired Three-Column Gallery](https://www.kaggle.com/datasets/khoaanh1234/vto-sd-lora-epoch12-paired-three-column-gallery)

Full unpaired holdout result dataset:

- Kaggle Dataset: [Epoch 12 Full Unpaired Holdout Evaluation](https://www.kaggle.com/datasets/khoaanh1234/vto-v2-epoch12-unpaired-full-eval-20260612)
- Kaggle input path: `/kaggle/input/datasets/khoaanh1234/vto-v2-epoch12-unpaired-full-eval-20260612`

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

## Model 1: Lightweight U-Net + GMM + TOM

The Model 1 notebook loads or produces separate best checkpoints for its U-Net,
GMM, and TOM modules. Final external Kaggle Dataset links have not been
recorded in this repository.
