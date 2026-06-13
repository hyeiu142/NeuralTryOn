# Model 1 Paired Test

The final report uses the common paired-test protocol:

```text
Source    clean_vto_dataset_test.csv
Pairing   cloth_id = person_id
Samples   996
Metrics   SSIM, PSNR, LPIPS-VGG
```

The current evaluator still preserves the completed legacy notebook behavior
and must be adjusted or filtered to this manifest before the confirmation run.

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
