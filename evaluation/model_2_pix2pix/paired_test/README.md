# Model 2 Paired Test

This workflow preserves the completed notebook evaluation protocol:

```text
Source    every image in VITON-HD/test/image
Pairing   cloth_id = person_id
Samples   2032
Metrics   SSIM, PSNR, LPIPS-VGG
```

Run:

```text
01_evaluate_metrics.py
02_generate_report.py
03_export_three_column_gallery.py
04_publish_kaggle_dataset.py   optional
```

The completed notebook reported:

```text
SSIM    0.8951
PSNR    21.49 dB
LPIPS   0.1145
```

The standalone evaluator adds resumable inference, per-sample metrics,
standard deviation, metric distributions, best/worst samples, and three-column
galleries.
