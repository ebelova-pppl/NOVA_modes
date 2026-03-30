from __future__ import annotations

import os
from pathlib import Path


def resolve_mode_csv_path(path_str: str, data_env: str = "NOVA_DATA") -> str:
    """
    Resolve a CSV path entry for a mode file.

    - Absolute paths are returned unchanged (normalized).
    - Relative paths are interpreted relative to $NOVA_DATA.
    """
    raw = path_str.strip()
    if not raw:
        raise ValueError("Empty mode path in CSV entry")

    path = Path(raw).expanduser()
    if path.is_absolute():
        return str(path)

    data_root = os.environ.get(data_env)
    if not data_root:
        raise RuntimeError(
            f"Relative mode path '{raw}' requires environment variable {data_env}."
        )

    return str((Path(data_root).expanduser() / path).resolve())
