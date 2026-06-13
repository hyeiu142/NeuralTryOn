"""Model-agnostic batch inference and artifact export."""

from __future__ import annotations

import csv
from pathlib import Path
from typing import Iterable

from src.models import TryOnModel


class InferenceRunner:
    """Generate deterministic result files and an output manifest."""

    def __init__(self, model: TryOnModel, output_dir: str | Path) -> None:
        self.model = model
        self.output_dir = Path(output_dir)
        self.image_dir = self.output_dir / "images"

    def run(self, samples: Iterable[dict]) -> Path:
        """Generate images for samples containing person_id and cloth_id."""
        self.image_dir.mkdir(parents=True, exist_ok=True)
        records = []
        for sample in samples:
            person_id = str(sample["person_id"])
            cloth_id = str(sample["cloth_id"])
            filename = f"{person_id}__{cloth_id}.jpg"
            self.model.predict(sample).convert("RGB").save(self.image_dir / filename, quality=95)
            records.append(
                {"person_id": person_id, "cloth_id": cloth_id, "result_image": filename}
            )
        manifest = self.output_dir / "inference_manifest.csv"
        with manifest.open("w", newline="", encoding="utf-8") as file:
            writer = csv.DictWriter(
                file,
                fieldnames=("person_id", "cloth_id", "result_image"),
            )
            writer.writeheader()
            writer.writerows(records)
        return manifest

