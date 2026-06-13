"""Package run outputs for transfer from Kaggle or another remote GPU."""

from pathlib import Path
from shutil import make_archive


def package_run(run_dir: str | Path, output: str | Path | None = None) -> Path:
    """Create a ZIP archive containing one complete experiment run."""
    run_dir = Path(run_dir).resolve()
    if not run_dir.is_dir():
        raise FileNotFoundError(f"Run directory not found: {run_dir}")
    output = Path(output).resolve() if output else run_dir.parent / run_dir.name
    archive = make_archive(str(output), "zip", root_dir=run_dir.parent, base_dir=run_dir.name)
    return Path(archive)

