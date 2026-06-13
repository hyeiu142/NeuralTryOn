"""Validate Model 2 classes and load the three evaluated checkpoints."""

from pathlib import Path

import torch


required_classes = ["GMM", "Stage1_SegmentationNet", "TOM_Generator_Stage2"]
missing_classes = [name for name in required_classes if name not in globals()]
if missing_classes:
    raise RuntimeError(
        "Hay chay cac cell dinh nghia kien truc Model 2 truoc evaluation. Thieu: "
        + ", ".join(missing_classes)
    )


def _find_checkpoint(filename):
    candidates = [
        Path("/kaggle/input/datasets/vnhttin/module-trained") / filename,
        Path("/kaggle/working") / filename,
    ]
    for candidate in candidates:
        if candidate.exists():
            return candidate
    matches = list(Path("/kaggle/input").rglob(filename))
    return matches[0] if matches else None


if "model_gmm" not in globals():
    model_gmm = GMM(DEVICE, img_size=(HEIGHT, WIDTH)).to(DEVICE)
if "model_stage1" not in globals():
    model_stage1 = Stage1_SegmentationNet().to(DEVICE)
if "model_tom" not in globals():
    model_tom = TOM_Generator_Stage2(in_channels=12).to(DEVICE)

for model, filename in [
    (model_gmm, "best_gmm.pth"),
    (model_stage1, "best_seg_stage1.pth"),
    (model_tom, "best_tom_stage2.pth"),
]:
    checkpoint = _find_checkpoint(filename)
    if checkpoint is None:
        raise FileNotFoundError(f"Khong tim thay checkpoint: {filename}")
    model.load_state_dict(torch.load(checkpoint, map_location=DEVICE))
    print(f"Loaded {filename}: {checkpoint}")

model_gmm.eval()
model_stage1.eval()
model_tom.eval()
processor = GeometricProcessor(HEIGHT, WIDTH) if "GeometricProcessor" in globals() else None
if processor is None:
    raise RuntimeError("Thieu GeometricProcessor tu notebook Model 2.")
print("Model 2 runtime ready.")
