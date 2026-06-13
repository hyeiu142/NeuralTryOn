# Model 2 Paired Test

The final report uses the common paired-test protocol:

```text
Source    clean_vto_dataset_test.csv
Pairing   cloth_id = person_id
Samples   996
Metrics   SSIM, PSNR, LPIPS-VGG
```

Run:

```text
01_evaluate_metrics.py
02_generate_report.py
03_export_three_column_gallery.py
04_publish_kaggle_dataset.py   optional
```

The completed legacy notebook reported the following values, which require
confirmation on the common 996-sample manifest:

```text
SSIM    0.8951
PSNR    21.49 dB
LPIPS   0.1145
```

The standalone evaluator adds resumable inference, per-sample metrics,
standard deviation, metric distributions, best/worst samples, and three-column
galleries.
