"""Manifest normalization and validation tests."""

import pandas as pd
import pytest

from src.data import VTONManifest


def test_manifest_normalizes_image_suffixes(tmp_path) -> None:
    path = tmp_path / "paired.csv"
    pd.DataFrame(
        {"person_id": ["00001_00.jpg"], "cloth_id": ["00001_00.jpg"]}
    ).to_csv(path, index=False)

    manifest = VTONManifest(path, "paired_test")
    pair = next(iter(manifest))

    assert pair.person_id == "00001_00"
    assert pair.is_paired
    assert manifest.validate(require_paired=True)["samples"] == 1


def test_manifest_rejects_unpaired_rows_when_required(tmp_path) -> None:
    path = tmp_path / "unpaired.csv"
    pd.DataFrame({"person_id": ["a"], "cloth_id": ["b"]}).to_csv(path, index=False)

    with pytest.raises(ValueError):
        VTONManifest(path, "paired_test").validate(require_paired=True)


def test_identity_override_creates_paired_reconstruction_rows(tmp_path) -> None:
    path = tmp_path / "source_pairs.csv"
    pd.DataFrame({"person_id": ["a"], "cloth_id": ["b"]}).to_csv(path, index=False)

    pair = next(iter(VTONManifest(path, "paired_test", pairing_mode="identity_override")))

    assert pair.person_id == pair.cloth_id == "a"
