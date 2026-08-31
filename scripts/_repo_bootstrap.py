"""Repository-relative import setup for directly executed scripts.

The project keeps shared Python modules in ``src/`` without installing them as
a site package.  Direct CLI entry points call :func:`ensure_repo_src_on_path`
before importing those modules so their behavior does not depend on a caller's
``PYTHONPATH``.
"""

from __future__ import annotations

import os
import sys
from pathlib import Path


def ensure_repo_src_on_path() -> Path:
    """Prepend this checkout's ``src`` directory and return the repo root."""

    repo_root = Path(__file__).resolve().parents[1]
    src_dir = repo_root / "src"
    if not src_dir.is_dir():
        raise RuntimeError(f"Repository src directory is missing: {src_dir}")
    src_text = str(src_dir)
    if src_text not in sys.path:
        sys.path.insert(0, src_text)
    return repo_root


def default_training_csv() -> Path:
    """Return ``$NOVA_TRAIN_CSV`` or this checkout's canonical label list."""

    configured = os.environ.get("NOVA_TRAIN_CSV")
    if configured:
        return Path(configured).expanduser()
    return (
        Path(__file__).resolve().parents[1]
        / "training_labels"
        / "tae_like_train.csv"
    )
