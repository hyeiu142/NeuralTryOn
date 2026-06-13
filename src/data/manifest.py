"""Validated CSV manifest access shared by training and evaluation."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Iterator

import pandas as pd


PERSON_COLUMNS = ("person_id", "person", "image_id", "id")
CLOTH_COLUMNS = ("cloth_id", "cloth", "garment_id", "id")


@dataclass(frozen=True)
class VTONPair:
    """One person-garment pair resolved from a clean manifest."""

    person_id: str
    cloth_id: str
    split: str

    @property
    def is_paired(self) -> bool:
        """Return whether the garment belongs to the target person image."""
        return self.person_id == self.cloth_id


class VTONManifest:
    """Load, normalize, validate, and iterate a VTON CSV manifest."""

    def __init__(self, path: str | Path, split: str, pairing_mode: str = "manifest") -> None:
        self.path = Path(path)
        self.split = split
        self.pairing_mode = pairing_mode
        if pairing_mode not in {"manifest", "identity", "identity_override"}:
            raise ValueError(f"Unsupported pairing mode: {pairing_mode}")
        if not self.path.exists():
            raise FileNotFoundError(f"Manifest not found: {self.path}")
        self.frame = pd.read_csv(self.path)
        self.person_column = _find_column(self.frame, PERSON_COLUMNS, "person")
        self.cloth_column = (
            self.person_column
            if pairing_mode == "identity"
            else _find_column(self.frame, CLOTH_COLUMNS, "cloth")
        )

    def __len__(self) -> int:
        return len(self.frame)

    def __iter__(self) -> Iterator[VTONPair]:
        for row in self.frame.itertuples(index=False):
            person_id = _normalize_id(getattr(row, self.person_column))
            cloth_id = _normalize_id(getattr(row, self.cloth_column))
            if self.pairing_mode == "identity_override":
                cloth_id = person_id
            yield VTONPair(
                person_id=person_id,
                cloth_id=cloth_id,
                split=self.split,
            )

    def validate(self, require_paired: bool | None = None) -> dict[str, int]:
        """Validate identifiers and optionally enforce paired or unpaired rows."""
        pairs = list(self)
        empty = sum(not pair.person_id or not pair.cloth_id for pair in pairs)
        paired = sum(pair.is_paired for pair in pairs)
        if empty:
            raise ValueError(f"{self.path} contains {empty} empty identifiers")
        if require_paired is True and paired != len(pairs):
            raise ValueError(f"{self.path} contains {len(pairs) - paired} unpaired rows")
        if require_paired is False and paired:
            raise ValueError(f"{self.path} contains {paired} paired rows")
        return {"samples": len(pairs), "paired": paired, "unpaired": len(pairs) - paired}


def _find_column(frame: pd.DataFrame, candidates: tuple[str, ...], role: str) -> str:
    for column in candidates:
        if column in frame.columns:
            return column
    raise ValueError(f"Manifest needs a {role} ID column; found {list(frame.columns)}")


def _normalize_id(value: object) -> str:
    value = str(value).strip()
    for suffix in (".jpg", ".jpeg", ".png"):
        if value.lower().endswith(suffix):
            value = value[: -len(suffix)]
            break
    return value
