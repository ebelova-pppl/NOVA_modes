from __future__ import annotations

import os
from pathlib import Path


def resolve_mode_csv_path(
    path_str: str,
    data_env: str = "NOVA_DATA",
    data_root: str | Path | None = None,
) -> str:
    """
    Resolve a CSV path entry for a mode file.

    - Absolute paths are returned unchanged (normalized).
    - Relative paths are interpreted relative to ``data_root`` when provided,
      otherwise relative to ``$NOVA_DATA``.
    """
    raw = path_str.strip()
    if not raw:
        raise ValueError("Empty mode path in CSV entry")

    path = Path(raw).expanduser()
    if path.is_absolute():
        return str(path)

    resolved_root = data_root if data_root is not None else os.environ.get(data_env)
    if not resolved_root:
        raise RuntimeError(
            f"Relative mode path '{raw}' requires environment variable {data_env}."
        )

    return str((Path(resolved_root).expanduser() / path).resolve())
