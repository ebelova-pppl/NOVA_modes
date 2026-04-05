# src/paths.py
from __future__ import annotations
import os
from pathlib import Path

def get_path(name: str) -> Path:
    value = os.environ.get(name)
    if not value:
        raise RuntimeError(f"Environment variable {name} is not set")
    return Path(value).expanduser().resolve()

NOVA_REPO = get_path("NOVA_REPO")                 # ~/src_nova
NOVA_DATA = get_path("NOVA_DATA")                 # global/cfs/cdirs/m314/nova/data
NOVA_MODELS = get_path("NOVA_MODELS")             # global/cfs/cdirs/m314/nova/models
NOVA_RESULTS = get_path("NOVA_RESULTS")           # global/cfs/cdirs/m314/nova/results
NOVA_RUN_RF = get_path("NOVA_RUN_RF")             # $SCRATCH/nova_rf
NOVA_RUN_CNN = get_path("NOVA_RUN_CNN")           # $SCRATCH/nova_cnn
NOVA_TRAIN_CSV = get_path("NOVA_TRAIN_CSV")       # ~/src_nova/training_labels/train_master.csv