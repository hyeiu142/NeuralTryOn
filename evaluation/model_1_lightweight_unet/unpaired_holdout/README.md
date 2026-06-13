# Model 1 Unpaired Holdout

This workflow preserves the Model 1 notebook's holdout protocol.

It reads all rows from `holdout_test.csv`, then shifts the CSV cloth list by one
position before pairing it with the original person list:

```python
shifted_cloths = cloths[1:] + cloths[:1]
```

Thus all 665 holdout people are evaluated using cross-garment pairs.

Run stages:

```text
01_generate_holdout.py
02_export_three_column_gallery.py
03_generate_report.py
04_publish_kaggle_dataset.py   optional
```

The qualitative gallery format is:

```text
Person Input | Target Cloth | Try-On Result
```

No SSIM, PSNR, or LPIPS is reported for unpaired holdout because there is no
pixel-aligned ground-truth image for the shifted person-cloth pair.
