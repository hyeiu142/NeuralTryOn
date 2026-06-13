# Reported Metrics

`reported_metrics.csv` records the completed results currently available for
Models 1 and 3.

These numbers are suitable for documenting project progress, but they are not
yet a fair ablation comparison:

```text
Model 1  2032 raw VITON-HD test images, LPIPS-VGG
Model 3   996 cleaned paired-test images, LPIPS-AlexNet
```

For the final three-model ranking, rerun every model using one shared manifest
and one shared LPIPS backbone.
