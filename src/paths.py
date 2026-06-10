# src/paths.py
from __future__ import annotations
import os
from pathlib import Path

def get_path(name: str) -> Path:
    value = os.environ.get(name)
    if not value:
        raise RuntimeError(f"Environment variable {name} is not set")
    return Path(value).expanduser().resolve()

def get_optional_path(name: str) -> Path | None:
    value = os.environ.get(name)
    if not value:
        return None
    return Path(value).expanduser().resolve()

NOVA_REPO = get_path("NOVA_REPO")                 # repository root
NOVA_TRAIN_CSV = get_path("NOVA_TRAIN_CSV")       # active default training CSV
NOVA_DATA = get_optional_path("NOVA_DATA")         # optional active NOVA data root
NOVA_MODELS = get_optional_path("NOVA_MODELS")     # optional shared model directory
NOVA_RESULTS = get_optional_path("NOVA_RESULTS")   # optional shared results directory
NOVA_RUN_RF = get_optional_path("NOVA_RUN_RF")     # optional scratch RF run directory
NOVA_RUN_CNN = get_optional_path("NOVA_RUN_CNN")   # optional scratch CNN run directory
