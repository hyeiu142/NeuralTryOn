# Model 1 Paired Test

This workflow preserves the completed Model 1 notebook protocol:

```text
Source    VITON-HD/test/image
Pairing   cloth_id = person_id
Samples   every image file found in test/image
Metrics   SSIM, PSNR, LPIPS-VGG
```

Run stages:

```text
01_evaluate_metrics.py
02_generate_report.py
03_export_three_column_gallery.py
04_publish_kaggle_dataset.py   optional
```

Outputs:

```text
/kaggle/working/model_1_lightweight_unet_evaluation/paired_test/
├── paired_test_manifest.csv
├── paired_test_metrics.csv
├── paired_test_summary.json
├── images/
├── metric_report/
└── comparisons_3_columns/
```

`01_evaluate_metrics.py` is resumable and saves progress after every sample.
The report contains metric distributions plus best/worst samples for error
analysis.
