# Dataset and Splits

All three models use the VITON-HD dataset:

```text
/kaggle/input/datasets/marquis03/high-resolution-viton-zalando-dataset
```

Inputs include person images, garments, garment masks, agnostic images, parsing
maps, DensePose, and OpenPose. Images are resized to `384 x 512`.

| Manifest | Samples | Purpose |
| --- | ---: | --- |
| `clean_vto_dataset_train.csv` | 9,520 | Training and validation |
| `clean_vto_dataset_test.csv` | 996 | Paired metrics with `cloth_id = person_id` |
| `holdout_test.csv` | 665 | Unpaired qualitative evaluation and error analysis |

Local CSV and caption copies live under `data/`, which is excluded from Git.
