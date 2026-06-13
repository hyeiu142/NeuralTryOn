"""Resolve local and Kaggle dataset locations without hard-coded notebook state."""

from __future__ import annotations

from pathlib import Path
from typing import Callable, Iterable


def find_dataset_root(
    candidates: Iterable[str | Path],
    validator: Callable[[Path], bool],
    search_root: str | Path = "/kaggle/input",
    max_depth: int = 4,
) -> Path:
    """Return the first valid explicit or discovered dataset directory."""
    for candidate in candidates:
        path = Path(candidate)
        if validator(path):
            return path

    root = Path(search_root)
    if root.exists():
        frontier = [root]
        for _ in range(max_depth + 1):
            next_frontier: list[Path] = []
            for path in frontier:
                if validator(path):
                    return path
                try:
                    next_frontier.extend(child for child in path.iterdir() if child.is_dir())
                except OSError:
                    continue
            frontier = next_frontier
    raise FileNotFoundError(f"No valid dataset root found under candidates or {root}")


def has_viton_hd_layout(path: Path) -> bool:
    """Check the minimum VITON-HD directory contract."""
    return (path / "train" / "image").is_dir() and (path / "test" / "image").is_dir()


def has_clean_manifest_layout(path: Path) -> bool:
    """Check the cleaned train, paired-test, and holdout manifests."""
    names = (
        "clean_vto_dataset_train.csv",
        "clean_vto_dataset_test.csv",
        "holdout_test.csv",
    )
    return all((path / name).is_file() for name in names)


def has_caption_layout(path: Path) -> bool:
    """Check the BLIP garment-caption directory contract."""
    return (path / "cloth-captions" / "train").is_dir() and (
        path / "cloth-captions" / "test"
    ).is_dir()
