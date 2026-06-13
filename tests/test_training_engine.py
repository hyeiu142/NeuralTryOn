"""Training engine integration test using a tiny CPU model."""

import torch
from torch.utils.data import DataLoader, TensorDataset

from src.tracking import RunTracker
from src.training import EarlyStopping, TrainingEngine


def test_training_engine_writes_best_checkpoint(tmp_path) -> None:
    inputs = torch.arange(0, 4, dtype=torch.float32).unsqueeze(1)
    targets = inputs * 2
    loader = DataLoader(TensorDataset(inputs, targets), batch_size=2)
    model = torch.nn.Linear(1, 1)
    optimizer = torch.optim.SGD(model.parameters(), lr=0.01)
    tracker = RunTracker(
        {
            "name": "tiny_train",
            "tracking": {
                "run_root": "experiments/runs",
                "registry": "experiments/registry.csv",
            },
        },
        tmp_path,
    )
    tracker.start()

    def step(current_model, batch, device):
        x, y = (tensor.to(device) for tensor in batch)
        return torch.nn.functional.mse_loss(current_model(x), y)

    engine = TrainingEngine(
        model,
        optimizer,
        torch.device("cpu"),
        tracker,
        step,
        step,
        early_stopping=EarlyStopping(patience=2),
    )
    summary = engine.fit(loader, loader, epochs=2)
    tracker.finish(summary)

    assert summary["best_epoch"] >= 1
    assert (tracker.run_dir / "checkpoints/best.pt").exists()

