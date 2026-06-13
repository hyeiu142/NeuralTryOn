# Model Adapter Migration Guide

The notebooks are preserved as immutable experiment records. A model becomes
fully script-executable after its architecture and preprocessing logic are
migrated behind the common production contracts.

## Required Adapter Contract

Each model adapter must implement `src.models.TryOnModel`:

```python
class ModelAdapter(TryOnModel):
    def load_checkpoint(self, checkpoint):
        ...

    def predict(self, sample):
        ...
```

Training adapters should provide:

1. A PyTorch model factory.
2. Dataset-to-batch preprocessing.
3. Model-specific train and validation step functions.
4. Optimizer and scheduler factories.
5. Checkpoint loading compatible with the completed notebook run.

The generic `TrainingEngine` handles the outer lifecycle: train/validation
epochs, tracking, scheduler steps, early stopping, and best checkpoints.

## Migration Acceptance Criteria

A migrated adapter is accepted only when:

1. It loads the existing published checkpoint.
2. It reproduces notebook preprocessing for a fixed sample.
3. Its inference output matches the notebook within a documented tolerance.
4. Paired evaluation runs through `PairedEvaluationRunner`.
5. Unpaired inference runs through `InferenceRunner`.
6. The registry status changes from `notebook_reference` to `production_adapter`.

Until these checks pass, the CLI intentionally keeps training, evaluation, and
inference in guarded dry-run mode and points to the completed notebook workflow.

