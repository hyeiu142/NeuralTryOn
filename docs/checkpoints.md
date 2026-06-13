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

Final checkpoint dataset:

- Kaggle Dataset: [Pix2Pix Final Models](https://www.kaggle.com/datasets/vnhttin/pix2pix-final-models)
- Kaggle input path: `/kaggle/input/datasets/vnhttin/pix2pix-final-models`

The final model consists of the GMM, Shape Generation, and Pix2Pix Stage 2
checkpoints.

## Model 1: Lightweight U-Net + GMM + TOM

Final checkpoint dataset:

- Kaggle Dataset: [Lightweight U-Net](https://www.kaggle.com/datasets/pannguyeen/lightweight-u-net)
- Kaggle input path: `/kaggle/input/datasets/pannguyeen/lightweight-u-net`

The final pipeline uses separate checkpoints for its U-Net, GMM, and TOM
modules.
