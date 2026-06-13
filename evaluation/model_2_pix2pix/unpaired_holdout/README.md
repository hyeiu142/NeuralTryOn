# Model 2 Unpaired Holdout

The Model 2 notebook loads `holdout_test.csv` and demonstrates several manual
person-cloth pairs. It has not yet run and saved the complete 665-sample
holdout.

This workflow completes that missing operational step while preserving the
notebook's original holdout pairs:

```text
person_id = holdout_test.csv person_id
cloth_id  = holdout_test.csv cloth_id
```

Run:

```text
01_generate_holdout.py
02_export_three_column_gallery.py
03_generate_report.py
04_publish_kaggle_dataset.py   optional
```

The output gallery format is:

```text
Person Input | Target Cloth | Try-On Result
```

No reconstruction metrics are calculated for unpaired holdout because it has
no pixel-aligned ground-truth try-on image.
