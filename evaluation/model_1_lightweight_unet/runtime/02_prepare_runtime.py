"""Validate the models and preprocessing objects created by the Model 1 notebook."""

import glob
from pathlib import Path

import torch


required_runtime = [
    "processor",
    "model_unet",
    "model_gmm",
    "TOM_Generator",
    "TPSGridGen",
]
missing_runtime = [name for name in required_runtime if name not in globals()]
if missing_runtime:
    raise RuntimeError(
        "Hay chay cac cell kien truc va load checkpoint trong notebook Model 1 "
        "truoc evaluation. Thieu: " + ", ".join(missing_runtime)
    )

model_unet = model_unet.to(DEVICE).eval()
model_gmm = model_gmm.to(DEVICE).eval()

if "model_tom" not in globals():
    candidates = [
        Path("/kaggle/working/tom_generator_ultimate.pth"),
        *[Path(path) for path in glob.glob("/kaggle/input/**/tom_generator_ultimate.pth", recursive=True)],
    ]
    tom_path = next((path for path in candidates if path.exists()), None)
    if tom_path is None:
        raise FileNotFoundError("Khong tim thay tom_generator_ultimate.pth.")
    model_tom = TOM_Generator(dropout_rate=0.0).to(DEVICE)
    model_tom.load_state_dict(torch.load(tom_path, map_location=DEVICE))

model_tom = model_tom.to(DEVICE).eval()
tps_generator = TPSGridGen(target_H=HEIGHT, target_W=WIDTH, grid_size=5).to(DEVICE)

print("Model 1 runtime ready:")
print(f"  U-Net : {type(model_unet).__name__}")
print(f"  GMM   : {type(model_gmm).__name__}")
print(f"  TOM   : {type(model_tom).__name__}")
