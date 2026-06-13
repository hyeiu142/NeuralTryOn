"""Check and install dependencies required by the Model 3 evaluation workflow."""

import importlib
import subprocess
import sys


PACKAGES = [
    ("numpy", "numpy"),
    ("scipy", "scipy"),
    ("diffusers", "diffusers==0.30.3"),
    ("transformers", "transformers==4.44.2"),
    ("peft", "peft==0.12.0"),
    ("accelerate", "accelerate==0.34.2"),
    ("xformers", "xformers"),
    ("cv2", "opencv-python-headless==4.9.0.80"),
    ("skimage", "scikit-image==0.24.0"),
    ("lpips", "lpips"),
]


def is_available(module_name):
    try:
        importlib.import_module(module_name)
        return True
    except ImportError:
        return False


missing = [
    package_name
    for module_name, package_name in PACKAGES
    if not is_available(module_name)
]

if not missing:
    print("Tat ca thu vien da san sang. Khong can cai them.")
else:
    print("Can cai them:", ", ".join(missing))
    try:
        subprocess.check_call(
            [
                sys.executable,
                "-m",
                "pip",
                "install",
                "-q",
                "--no-input",
                *missing,
            ]
        )
    except subprocess.CalledProcessError as exc:
        raise RuntimeError(
            "Khong tai duoc package. Hay bat Internet trong Kaggle Settings "
            "hoac restart session de khoi phuc cac thu vien mac dinh, sau do chay lai Cell 1."
        ) from exc

print("Install check done.")
print("Neu vua cai package moi, restart session roi chay Cell 12 tro di.")
