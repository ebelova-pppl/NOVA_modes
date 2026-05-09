# src/paths.py
from __future__ import annotations
import os
from pathlib import Path

def get_path(name: str) -> Path:
    value = os.environ.get(name)
    if not value:
        raise RuntimeError(f"Environment variable {name} is not set")
    return Path(value).expanduser().resolve()

NOVA_REPO = get_path("NOVA_REPO")                 # repository root
NOVA_DATA = get_path("NOVA_DATA")                 # active NOVA data root
NOVA_MODELS = get_path("NOVA_MODELS")             # active shared model directory
NOVA_RESULTS = get_path("NOVA_RESULTS")           # active shared results directory
NOVA_RUN_RF = get_path("NOVA_RUN_RF")             # scratch RF run directory
NOVA_RUN_CNN = get_path("NOVA_RUN_CNN")           # scratch CNN run directory
NOVA_TRAIN_CSV = get_path("NOVA_TRAIN_CSV")       # active default training CSV
