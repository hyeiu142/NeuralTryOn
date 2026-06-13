"""Evaluation entry point and production-adapter readiness check."""

import argparse
import sys
from pathlib import Path

import lpips
import torch
from PIL import Image

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from src.config import load_experiment_config
from src.data import VTONManifest
from src.data.paths import find_dataset_root, has_caption_layout, has_clean_manifest_layout, has_viton_hd_layout
from src.evaluation import PairedEvaluationRunner
from src.models import get_model_spec
from src.models.model_3_sd_lora.inference import Model3InferenceAdapter


def main() -> None:
    """Validate evaluation configuration or dispatch a migrated adapter."""
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True, type=Path)
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--checkpoint", type=Path)
    parser.add_argument("--output-dir", type=Path, default=Path("/kaggle/working/model_3_paired_evaluation"))
    parser.add_argument("--max-samples", type=int)
    parser.add_argument("--viton-root", type=Path)
    parser.add_argument("--csv-root", type=Path)
    parser.add_argument("--caption-root", type=Path)
    args = parser.parse_args()

    config = load_experiment_config(args.config)
    spec = get_model_spec(config["model"]["key"])
    if args.dry_run:
        print(f"Evaluation request valid: {config['name']}")
        print(f"Metrics: {', '.join(config['evaluation']['metrics'])}")
        print(f"Existing workflow: {spec.evaluation_package}")
        return
    if config["model"]["key"] == "model_3":
        if args.checkpoint is None:
            parser.error("Model 3 evaluation requires --checkpoint")
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
        manifest = VTONManifest(
            Path(csv_root) / "clean_vto_dataset_test.csv",
            "paired_test",
            pairing_mode="identity_override",
        )
        samples = list(manifest)
        if args.max_samples is not None:
            samples = samples[: args.max_samples]

        def predict(pair):
            return adapter.predict({"person_id": pair.person_id, "cloth_id": pair.cloth_id})

        def target(pair):
            image = Image.open(Path(viton_root) / "test/image" / f"{pair.person_id}.jpg").convert("RGB")
            return adapter.preprocessor._crop_resize(image)

        device = "cuda" if torch.cuda.is_available() else "cpu"
        runner = PairedEvaluationRunner(
            args.output_dir,
            lpips_model=lpips.LPIPS(net="alex").to(device).eval(),
            device=device,
        )
        print(runner.run(samples, predict, target))
        return
    raise RuntimeError(
        f"Use the completed workflow under {spec.evaluation_package} until the "
        f"{spec.display_name} production adapter is migrated."
    )


if __name__ == "__main__":
    main()
