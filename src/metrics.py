"""Image reconstruction metrics used by all three model pipelines."""

from collections.abc import Mapping

import numpy as np
import torch
from skimage.metrics import peak_signal_noise_ratio, structural_similarity


def reconstruction_metrics(
    ground_truth: np.ndarray,
    prediction: np.ndarray,
    lpips_model: torch.nn.Module | None = None,
    device: str | torch.device = "cpu",
) -> dict[str, float]:
    """Compute full-image SSIM, PSNR, and optionally LPIPS.

    Both images must be RGB arrays in the [0, 1] range.
    """
    ground_truth = np.asarray(ground_truth, dtype=np.float32)
    prediction = np.asarray(prediction, dtype=np.float32)
    if ground_truth.shape != prediction.shape:
        raise ValueError(
            f"Image shapes must match: {ground_truth.shape} != {prediction.shape}"
        )

    metrics = {
        "ssim": float(
            structural_similarity(
                ground_truth,
                prediction,
                data_range=1.0,
                channel_axis=2,
            )
        ),
        "psnr": float(
            peak_signal_noise_ratio(ground_truth, prediction, data_range=1.0)
        ),
    }
    if lpips_model is not None:
        gt_tensor = _image_to_tensor(ground_truth, device)
        pred_tensor = _image_to_tensor(prediction, device)
        with torch.no_grad():
            metrics["lpips"] = float(
                lpips_model(gt_tensor, pred_tensor).item()
            )
    return metrics


def summarize_metrics(
    records: list[Mapping[str, float]],
    metric_names: tuple[str, ...] = ("ssim", "psnr", "lpips"),
) -> dict[str, dict[str, float]]:
    """Return mean and standard deviation for available metric columns."""
    summary: dict[str, dict[str, float]] = {"mean": {}, "std": {}}
    for name in metric_names:
        values = np.asarray(
            [record[name] for record in records if name in record],
            dtype=np.float64,
        )
        if values.size:
            summary["mean"][name] = float(values.mean())
            summary["std"][name] = float(values.std(ddof=1)) if values.size > 1 else 0.0
    return summary


def _image_to_tensor(image: np.ndarray, device: str | torch.device) -> torch.Tensor:
    tensor = torch.from_numpy(image).permute(2, 0, 1).unsqueeze(0).float()
    return (tensor.to(device) * 2.0) - 1.0
