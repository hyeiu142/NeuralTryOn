"""Check dependencies required by the Model 2 evaluation workflow."""

import importlib
import subprocess
import sys


PACKAGES = [
    ("numpy", "numpy"),
    ("pandas", "pandas"),
    ("skimage", "scikit-image"),
    ("lpips", "lpips"),
    ("matplotlib", "matplotlib"),
]
missing = []
for module_name, package_name in PACKAGES:
    try:
        importlib.import_module(module_name)
    except ImportError:
        missing.append(package_name)
if missing:
    subprocess.check_call([sys.executable, "-m", "pip", "install", "-q", "--no-input", *missing])
print("Model 2 evaluation dependencies are ready.")
