# Reported Metrics

`reported_metrics.csv` records the results currently available for all three
models. Model 2 has completed paired evaluation, while its full holdout export
is still pending.

The final report defines one shared paired-test protocol:

```text
Model 1  996 cleaned paired-test images, LPIPS-VGG
Model 2  996 cleaned paired-test images, LPIPS-VGG
Model 3  996 cleaned paired-test images, LPIPS-AlexNet
```

Model 1 and Model 2 values are retained from completed legacy runs and require
confirmation on the shared 996-sample manifest. A strict LPIPS ranking also
requires one shared LPIPS backbone.
