# Dataset and Splits

## Image Dataset

All three models use VITON-HD:

```text
/kaggle/input/datasets/marquis03/high-resolution-viton-zalando-dataset
```

Important components include person images, garments, garment masks, agnostic
images, parsing maps, DensePose, and OpenPose.

## Cleaned CSV Splits

```text
clean_vto_dataset_train.csv  9,520 samples
clean_vto_dataset_test.csv     996 samples
holdout_test.csv               665 samples
```

| Split | Role |
| --- | --- |
| Train | Model optimization |
| Test | Paired reconstruction by forcing `cloth_id = person_id`; SSIM, PSNR, LPIPS |
| Holdout | Unpaired try-on using the original person-cloth pairs |

Models 1 and 2 create validation subsets from the cleaned training manifest.
Model 3 uses `clean_vto_dataset_test.csv` as its validation loader during
training and later reports paired metrics on the same manifest. Its paired
result must therefore be described as validation-seen rather than performance
on a completely untouched test set.

Holdout is reserved for practical cross-garment comparison and error analysis
because unpaired examples do not have pixel-aligned ground truth.

Local CSV and caption copies live under `data/`, which is excluded from Git.
