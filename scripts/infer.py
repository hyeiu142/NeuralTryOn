"""Inference entry point and production-adapter readiness check."""

import argparse
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from src.config import load_experiment_config
from src.data.paths import find_dataset_root, has_caption_layout, has_clean_manifest_layout, has_viton_hd_layout
from src.models import get_model_spec
from src.models.model_3_sd_lora.inference import Model3InferenceAdapter


def main() -> None:
    """Validate inference configuration or dispatch a migrated adapter."""
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True, type=Path)
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--checkpoint", type=Path)
    parser.add_argument("--person-id")
    parser.add_argument("--cloth-id")
    parser.add_argument("--split", default="test")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--output", type=Path, default=Path("/kaggle/working/model_3_result.jpg"))
    parser.add_argument("--viton-root", type=Path)
    parser.add_argument("--csv-root", type=Path)
    parser.add_argument("--caption-root", type=Path)
    args = parser.parse_args()

    config = load_experiment_config(args.config)
    spec = get_model_spec(config["model"]["key"])
    if args.dry_run:
        print(f"Inference request valid: {config['name']}")
        print(f"Unpaired inference supported: {spec.supports_unpaired_inference}")
        print(f"Existing workflow: {spec.evaluation_package}/unpaired_holdout")
        return
    if config["model"]["key"] == "model_3":
        required = (args.checkpoint, args.person_id, args.cloth_id)
        if any(value is None for value in required):
            parser.error("Model 3 inference requires --checkpoint, --person-id, and --cloth-id")
        viton_root = args.viton_root or find_dataset_root(
            ["/kaggle/input/datasets/marquis03/high-resolution-viton-zalando-dataset", PROJECT_ROOT / "data/VITON-HD"],
            has_viton_hd_layout,
        )
        csv_root = args.csv_root or find_dataset_root(
            ["/kaggle/input/datasets/cthnhoddt/dlp-cleandatacsv", PROJECT_ROOT / "data/DLP_CleanDataCSV"],
            has_clean_manifest_layout,
        )
        caption_root = args.caption_root or find_dataset_root(
            ["/kaggle/input/datasets/cthnhoddt/dlp-cloth-caption", PROJECT_ROOT / "data/DLP_Cloth_Caption"],
            has_caption_layout,
        )
        adapter = Model3InferenceAdapter(config, viton_root, csv_root, caption_root)
        adapter.load_checkpoint(args.checkpoint)
        result = adapter.predict(
            {"person_id": args.person_id, "cloth_id": args.cloth_id, "split": args.split, "seed": args.seed}
        )
        args.output.parent.mkdir(parents=True, exist_ok=True)
        result.save(args.output, quality=95)
        print(args.output)
        return
    raise RuntimeError(
        f"Use {spec.evaluation_package}/unpaired_holdout until the "
        f"{spec.display_name} production adapter is migrated."
    )


if __name__ == "__main__":
    main()
