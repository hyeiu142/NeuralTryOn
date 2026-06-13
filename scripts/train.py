"""Production training entry point and migration guard."""

import argparse
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from src.config import load_experiment_config
from src.data.paths import (
    find_dataset_root,
    has_caption_layout,
    has_clean_manifest_layout,
    has_viton_hd_layout,
)
from src.models import get_model_spec
from src.models.model_3_sd_lora.trainer import train_model3
from src.runtime import inspect_runtime


def main() -> None:
    """Validate a training request before dispatching a production adapter."""
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True, type=Path)
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--viton-root", type=Path)
    parser.add_argument("--csv-root", type=Path)
    parser.add_argument("--caption-root", type=Path)
    parser.add_argument("--output-dir", type=Path, default=Path("/kaggle/working/vto_v2_production"))
    parser.add_argument("--resume-dir", type=Path)
    parser.add_argument("--epochs", type=int)
    parser.add_argument("--max-train-samples", type=int)
    parser.add_argument("--max-validation-samples", type=int)
    parser.add_argument("--wandb", action="store_true")
    args = parser.parse_args()

    config = load_experiment_config(args.config)
    spec = get_model_spec(config["model"]["key"])
    report = inspect_runtime(config, PROJECT_ROOT)
    if args.dry_run:
        print(f"Training request valid: {config['name']}")
        print(f"Model: {spec.display_name}")
        print(f"CUDA available: {report['cuda_available']}")
        print(f"Adapter status: {spec.production_adapter_status}")
        return

    if config["model"]["key"] == "model_3":
        viton_root = args.viton_root or find_dataset_root(
            [
                "/kaggle/input/datasets/marquis03/high-resolution-viton-zalando-dataset",
                PROJECT_ROOT / "data/VITON-HD",
            ],
            has_viton_hd_layout,
        )
        csv_root = args.csv_root or find_dataset_root(
            [
                "/kaggle/input/datasets/cthnhoddt/dlp-cleandatacsv",
                PROJECT_ROOT / "data/DLP_CleanDataCSV",
            ],
            has_clean_manifest_layout,
        )
        caption_root = args.caption_root or find_dataset_root(
            [
                "/kaggle/input/datasets/cthnhoddt/dlp-cloth-caption",
                PROJECT_ROOT / "data/DLP_Cloth_Caption",
            ],
            has_caption_layout,
        )
        summary = train_model3(
            config,
            viton_root=viton_root,
            csv_root=csv_root,
            caption_root=caption_root,
            output_dir=args.output_dir,
            max_train_samples=args.max_train_samples,
            max_validation_samples=args.max_validation_samples,
            epochs=args.epochs,
            resume_dir=args.resume_dir,
            use_wandb=args.wandb,
        )
        print(f"Training complete: {summary}")
        return

    raise RuntimeError(
        f"{spec.display_name} training is currently executed by "
        f"{spec.source_notebook}. Its production adapter has not been migrated, "
        "so refusing to start a different implementation."
    )


if __name__ == "__main__":
    main()
