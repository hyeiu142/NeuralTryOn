"""Framework-neutral contract implemented by production model adapters."""

from __future__ import annotations

from abc import ABC, abstractmethod
from pathlib import Path
from typing import Any

from PIL import Image


class TryOnModel(ABC):
    """Common inference and checkpoint contract for all VTO models."""

    @abstractmethod
    def load_checkpoint(self, checkpoint: str | Path) -> None:
        """Load model weights required for inference."""

    @abstractmethod
    def predict(self, sample: dict[str, Any]) -> Image.Image:
        """Generate one RGB try-on result."""

    def trainable_parameters(self) -> tuple[int, int] | None:
        """Return trainable and total parameter counts when available."""
        return None

