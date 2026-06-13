# Configurations

Configuration is separated by responsibility:

```text
configs/
├── data/          Shared dataset and split definitions
├── models/        Architecture-only settings
├── experiments/   Training, evaluation, and tracking settings
└── tracking/      Shared experiment-tracking defaults
```

An experiment configuration references one data configuration, one model
configuration, and one tracking configuration. `src.config.load_experiment_config`
resolves these references into one reproducible configuration snapshot.

The existing notebooks remain self-contained and are not modified to consume
these files. The configurations currently document their completed experiments
and provide the standard interface for future scripts.

The configurations cover all three models, including their stage-specific
architectures, data loaders, augmentation, optimization, losses, checkpoint
contracts, and evaluation settings. See
[`docs/configuration_reference.md`](../docs/configuration_reference.md) for the
field-level organization and notebook mapping.
