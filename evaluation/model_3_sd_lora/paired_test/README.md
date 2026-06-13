# Paired Test Evaluation

This workflow evaluates reconstruction quality on
`clean_vto_dataset_test.csv`.

For every test `person_id`:

```text
person_id = CSV person_id
cloth_id  = person_id
GT        = VITON-HD/test/image/{person_id}.jpg
```

Stages:

```text
01_evaluate_metrics.py             Full resumable inference and metrics
02_export_three_column_gallery.py  Person | Target Cloth | Try-On Result
03_publish_kaggle_dataset.py       Optional Kaggle Dataset publishing
```

Per-sample CSV columns:

```text
checkpoint, sample_index, person_id, cloth_id, seed,
ssim, psnr, lpips, seconds, status, error
```

Main output:

```text
/kaggle/working/infer_results_v2_epoch12/paired_test_metrics/
```
