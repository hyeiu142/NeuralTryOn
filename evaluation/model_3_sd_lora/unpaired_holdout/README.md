# Unpaired Holdout Evaluation

This workflow evaluates practical cross-garment try-on using the original pairs
from `holdout_test.csv`.

Stages:

```text
01_evaluate_holdout.py       Resumable full holdout inference
02_generate_report.py        Ranking, summaries, best/worst galleries
03_publish_kaggle_dataset.py Optional Kaggle Dataset publishing
```

There is no pixel-aligned ground truth for unpaired pairs. The workflow reports:

```text
clip_garment_similarity
outside_mae
raw_outside_mae
mask_area
seconds
```

It also saves result, raw model output, and five-panel comparison images.

Completed full holdout artifact:

[Epoch 12 Full Unpaired Holdout Evaluation](https://www.kaggle.com/datasets/khoaanh1234/vto-v2-epoch12-unpaired-full-eval-20260612)
