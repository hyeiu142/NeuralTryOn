# Common Model Comparison

This directory creates report-ready tables and charts from completed model
results.

Run:

```bash
python evaluation/common_comparison/01_compare_reported_metrics.py
```

Input:

```text
results/metrics/reported_metrics.csv
```

Outputs:

```text
results/metrics/reported_metrics_table.md
results/metrics/reported_metrics_comparison.png
```

The current table contains Models 1 and 3. Model 2 can be added as a new CSV
row after its evaluation is complete.

The generated chart is labeled as a reported-results overview because Models 1
and 3 currently use different paired manifests and LPIPS backbones.
