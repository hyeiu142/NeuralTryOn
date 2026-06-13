"""Configuration contract tests."""

from pathlib import Path

from src.config import load_experiment_config


ROOT = Path(__file__).resolve().parents[1]


def test_all_experiment_configs_resolve() -> None:
    for path in (ROOT / "configs/experiments").glob("*.yaml"):
        config = load_experiment_config(path)
        assert config["model"]["key"] in {"model_1", "model_2", "model_3"}
        assert config["training"]["seed"] == 42
        assert config["evaluation"]["metrics"] == ["ssim", "psnr", "lpips"]

