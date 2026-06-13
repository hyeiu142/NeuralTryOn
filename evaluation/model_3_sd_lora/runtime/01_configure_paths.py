"""Resolve Kaggle inputs, validate checkpoint metadata, and configure runtime."""

import json
import shutil
from pathlib import Path

import torch


# ── Checkpoint / output ──────────────────────────────────────────
CHECKPOINT_EPOCH = 12
CHECKPOINT_NAME = f"checkpoint_epoch_{CHECKPOINT_EPOCH:02d}"
CHECKPOINT_SOURCE = Path("/kaggle/input/datasets/khoaanh1234/ckpt-epoch-12-yen")
CKPT_DIR = Path("/kaggle/working/checkpoint_latest")
OUTPUT_DIR = Path(f"/kaggle/working/infer_results_v2_epoch{CHECKPOINT_EPOCH:02d}")
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)


def _has_viton_structure(path):
    path = Path(path)
    return (path / "train" / "image").exists() and (path / "test" / "image").exists()


def _has_csv_structure(path):
    path = Path(path)
    return (
        (path / "clean_vto_dataset_train.csv").exists()
        and (path / "clean_vto_dataset_test.csv").exists()
        and (path / "holdout_test.csv").exists()
    )


def _has_caption_structure(path):
    return (Path(path) / "cloth-captions").exists()


def _checkpoint_dir(path):
    path = Path(path)
    return path if path.name == "checkpoint_latest" else path / "checkpoint_latest"


def _has_checkpoint_structure(path):
    ckpt = _checkpoint_dir(path)
    return (
        (ckpt / "unet_lora").exists()
        and (ckpt / "conv_in.pt").exists()
        and (ckpt / "perceiver.pt").exists()
        and (ckpt / "cloth_spatial.pt").exists()
    )


def _checkpoint_missing_items(path):
    ckpt = _checkpoint_dir(path)
    required = [
        "unet_lora",
        "conv_in.pt",
        "perceiver.pt",
        "cloth_spatial.pt",
    ]
    return [item for item in required if not (ckpt / item).exists()]


def _print_checkpoint_debug(path):
    path = Path(path)
    print(f"  Checkpoint exists: {path.exists()} {path}")
    if path.exists():
        try:
            print("  Checkpoint root content:")
            for p in sorted(path.iterdir())[:20]:
                kind = "DIR " if p.is_dir() else "FILE"
                print(f"    {kind} {p.name}")
        except Exception as e:
            print(f"    Khong doc duoc root: {e}")
    ckpt = _checkpoint_dir(path)
    print(f"  Expected checkpoint_latest: {ckpt}")
    print(f"  checkpoint_latest exists: {ckpt.exists()}")
    if ckpt.exists():
        try:
            print("  checkpoint_latest content:")
            for p in sorted(ckpt.iterdir())[:20]:
                kind = "DIR " if p.is_dir() else "FILE"
                print(f"    {kind} {p.name}")
        except Exception as e:
            print(f"    Khong doc duoc checkpoint_latest: {e}")
    missing = _checkpoint_missing_items(path)
    if missing:
        print(f"  Missing inside checkpoint_latest: {missing}")


def _resolve_checkpoint_source(path):
    path = Path(path)
    roots = [
        path,
        Path("/kaggle/input/ckpt-epoch-12-yen"),
    ]
    for root in roots:
        if _has_checkpoint_structure(root):
            return root
        if root.exists():
            found = _find_valid_dir_under(root, _has_checkpoint_structure, max_depth=3)
            if found:
                return found
    return None


def _first_valid(paths, validator):
    for raw_path in paths:
        path = Path(raw_path)
        if path.exists() and validator(path):
            return path
    return None


def _find_valid_dir_under(root, validator, max_depth=4):
    root = Path(root)
    if not root.exists():
        return None
    candidates = [root]
    frontier = [root]
    for _ in range(max_depth):
        next_frontier = []
        for base in frontier:
            try:
                children = [p for p in base.iterdir() if p.is_dir()]
            except Exception:
                children = []
            candidates.extend(children)
            next_frontier.extend(children)
        frontier = next_frontier

    for path in sorted(set(candidates)):
        if validator(path):
            return path
    return None


# ── Locate datasets/checkpoint ───────────────────────────────────
VITON_ROOT = _first_valid(
    [
        "/kaggle/input/datasets/marquis03/high-resolution-viton-zalando-dataset",
        "/kaggle/input/VITON-HD",
        "/kaggle/input/viton-hd",
        "/home/yennguyen/VTO/data/VITON-HD",
    ],
    _has_viton_structure,
)
if VITON_ROOT is None:
    VITON_ROOT = _find_valid_dir_under("/kaggle/input", _has_viton_structure)

CSV_ROOT = _first_valid(
    [
        "/kaggle/input/datasets/cthnhoddt/dlp-cleandatacsv",
        "/kaggle/input/DLP_CleanDataCSV",
        "/kaggle/input/dlp-cleandatacsv",
        "/home/yennguyen/VTO/data/DLP_CleanDataCSV",
    ],
    _has_csv_structure,
)
if CSV_ROOT is None:
    CSV_ROOT = _find_valid_dir_under("/kaggle/input", _has_csv_structure)

CAPTION_ROOT = _first_valid(
    [
        "/kaggle/input/datasets/cthnhoddt/dlp-cloth-caption",
        "/kaggle/input/DLP_Cloth_Caption",
        "/kaggle/input/dlp-cloth-caption",
        "/home/yennguyen/VTO/data/DLP_Cloth_Caption",
    ],
    _has_caption_structure,
)
if CAPTION_ROOT is None:
    CAPTION_ROOT = _find_valid_dir_under("/kaggle/input", _has_caption_structure)

print("Kiem tra duong dan:")
RESOLVED_CHECKPOINT_SOURCE = _resolve_checkpoint_source(CHECKPOINT_SOURCE)
print(f"  Checkpoint src : {'OK' if RESOLVED_CHECKPOINT_SOURCE else 'MISSING'} {CHECKPOINT_SOURCE}")
if RESOLVED_CHECKPOINT_SOURCE and RESOLVED_CHECKPOINT_SOURCE != CHECKPOINT_SOURCE:
    print(f"  Checkpoint used: {RESOLVED_CHECKPOINT_SOURCE}")
if RESOLVED_CHECKPOINT_SOURCE is None:
    _print_checkpoint_debug(CHECKPOINT_SOURCE)
print(f"  VITON-HD root  : {'OK' if VITON_ROOT else 'MISSING'} {VITON_ROOT}")
print(f"  CSV root       : {'OK' if CSV_ROOT else 'MISSING'} {CSV_ROOT}")
print(f"  Caption root   : {'OK' if CAPTION_ROOT else 'MISSING'} {CAPTION_ROOT}")

missing = []
if RESOLVED_CHECKPOINT_SOURCE is None:
    missing.append(str(CHECKPOINT_SOURCE))
if VITON_ROOT is None:
    missing.append("VITON-HD root")
if CSV_ROOT is None:
    missing.append("DLP_CleanDataCSV co holdout_test.csv")
if CAPTION_ROOT is None:
    missing.append("DLP_Cloth_Caption")
if missing:
    raise FileNotFoundError("Khong tim thay: " + ", ".join(missing))


# ── Prepare checkpoint ───────────────────────────────────────────
if CKPT_DIR.exists():
    shutil.rmtree(CKPT_DIR)
old_history = CKPT_DIR.parent / "loss_history.json"
if old_history.exists():
    old_history.unlink()

src = Path(RESOLVED_CHECKPOINT_SOURCE)
print(f"\nChuan bi checkpoint tu: {src}")
src_ckpt = _checkpoint_dir(src)
shutil.copytree(src_ckpt, CKPT_DIR)
hist_src = src / "loss_history.json"
if hist_src.exists():
    shutil.copy2(hist_src, CKPT_DIR.parent / "loss_history.json")

if not _has_checkpoint_structure(CKPT_DIR):
    raise FileNotFoundError(f"Checkpoint khong dung cau truc: {CKPT_DIR}")

print("Checkpoint da san sang.")


# ── Show checkpoint content ──────────────────────────────────────
print("\nNoi dung checkpoint_latest:")
for path in sorted(CKPT_DIR.rglob("*"))[:32]:
    size = f"({path.stat().st_size / 1024 / 1024:.1f} MB)" if path.is_file() else ""
    kind = "FILE" if path.is_file() else "DIR "
    print(f"  {kind} {path.relative_to(CKPT_DIR)} {size}")

for hist_path in [CKPT_DIR.parent / "loss_history.json", src.parent / "loss_history.json"]:
    if hist_path.exists():
        with open(hist_path, "r", encoding="utf-8") as f:
            hist = json.load(f)
        if int(hist.get("epochs_done", -1)) != CHECKPOINT_EPOCH:
            raise ValueError(
                f"Checkpoint metadata sai epoch: epochs_done={hist.get('epochs_done')} "
                f"nhung CHECKPOINT_EPOCH={CHECKPOINT_EPOCH}"
            )
        cfg_meta = hist.get("config", {})
        expected_meta = {
            "unet_in_channels": "17",
            "lora_rank": "16",
            "lora_alpha": "16",
            "num_image_tokens": "8",
            "perceiver_depth": "2",
            "perceiver_heads": "8",
            "trigger_word": "hyeiu_cloth",
        }
        mismatches = [
            f"{k}: checkpoint={cfg_meta.get(k)} expected={v}"
            for k, v in expected_meta.items()
            if k in cfg_meta and str(cfg_meta.get(k)) != v
        ]
        if mismatches:
            raise ValueError("Checkpoint config khong khop voi infer:\n  " + "\n  ".join(mismatches))
        print("\nCheckpoint metadata:")
        print(f"  epochs_done   : {hist.get('epochs_done')}")
        print(f"  global_step   : {hist.get('global_step')}")
        print(f"  best_val_loss : {hist.get('best_val_loss')}")
        print(f"  val_losses    : {hist.get('epoch_val_losses')}")
        break


# ── Model/runtime config: must match v2 17ch training ─────────────
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
DTYPE = torch.float16
MODEL_ID = "runwayml/stable-diffusion-inpainting"
IP_ADAPTER = "h94/IP-Adapter"

H, W = 512, 384
NUM_STEPS = 45
CFG_SCALE = 1.3
CFG_SWEEP = [1.0, 1.3, 1.5, 2.0]
TRIGGER = "hyeiu_cloth"
INFER_CSV_NAME = "holdout_test.csv"

USE_EMA = False

LORA_RANK = 16
LORA_ALPHA = 16
LORA_DROPOUT = 0.05
LORA_TARGETS = ["to_q", "to_k", "to_v", "to_out.0", "ff.net.0.proj", "ff.net.2"]

NUM_IMAGE_TOKENS = 8
NUM_SPATIAL_TOKENS = 64
PERCEIVER_DEPTH = 2
PERCEIVER_HEADS = 8
UNET_IN_CHANNELS = 17

print("\nConfig:")
print(f"  Device    : {DEVICE}")
print(f"  DType     : {DTYPE}")
print(f"  Size      : {H}x{W}")
print(f"  Steps     : {NUM_STEPS}")
print(f"  CFG       : {CFG_SCALE}")
print(f"  Perceiver : tokens={NUM_IMAGE_TOKENS}, depth={PERCEIVER_DEPTH}")
print(f"  UNet input: {UNET_IN_CHANNELS} channels")
print(f"  Infer CSV : {CSV_ROOT / INFER_CSV_NAME}")
print(f"  USE_EMA   : {USE_EMA}")
print("\nConfig xong.")
