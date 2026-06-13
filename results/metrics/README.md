# Reported Metrics

`reported_metrics.csv` records the results currently available for all three
models. Model 2 has completed paired evaluation, while its full holdout export
is still pending.

These numbers are suitable for documenting project progress, but they are not
yet a fair ablation comparison:

```text
Model 1  2032 raw VITON-HD test images, LPIPS-VGG
Model 2  2032 raw VITON-HD test images, LPIPS-VGG
Model 3   996 cleaned paired-test images, LPIPS-AlexNet
```

For the final three-model ranking, rerun every model using one shared manifest
and one shared LPIPS backbone.
