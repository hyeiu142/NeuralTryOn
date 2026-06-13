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

The current table contains reported paired results for all three models. Model
2 is labeled as paired-complete because its full holdout export is still
pending.

The generated chart is labeled as a reported-results overview because Models 1
and 3 currently use different paired manifests and LPIPS backbones.
