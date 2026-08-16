#!/usr/bin/env python3
"""Preprocess one NOVA shot into deterministic TAE-side and EAE-side lists.

Example:
    python scripts/make_tae_like_list.py --shot_dir /path/to/SHOT \
        --out_dir /path/to/output
"""

from __future__ import annotations

import argparse
import math
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np


REPO_ROOT = Path(__file__).resolve().parents[1]
SRC_DIR = REPO_ROOT / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from cont_features import load_datcon_for_mode  # noqa: E402
from mode_features import radial_centroid, radial_width  # noqa: E402
from nova_mode_loader import load_mode_from_nova  # noqa: E402
from tae_eae_features import upper2_scalars  # noqa: E402
from tae_rule_io import (  # noqa: E402
    RULE_OUTPUT_FIELDS,
    datcon_path_for_mode,
    empty_rule_row,
    input_fingerprint,
    portable_mode_key,
    rule_row_sort_key,
    stable_json,
    write_dict_csv,
)


DEFAULT_FRACTION_TAE_THRESHOLD = 0.5
DEFAULT_FRACTION_EAE_THRESHOLD = 0.4
DEFAULT_SIGNED_DELTA_EAE_THRESHOLD = -0.1


@dataclass(frozen=True)
class PreprocessResult:
    """In-memory result returned by :func:`preprocess_shot`."""

    shot_dir: Path
    shot: str
    rows: tuple[dict[str, Any], ...]

    @property
    def tae_rows(self) -> list[dict[str, Any]]:
        return [
            row
            for row in self.rows
            if row.get("processing_status") == "READY_FOR_RULES"
        ]

    @property
    def eae_rows(self) -> list[dict[str, Any]]:
        return [
            row
            for row in self.rows
            if row.get("processing_status") == "ROUTED_EAE"
        ]

    @property
    def invalid_rows(self) -> list[dict[str, Any]]:
        return [
            row for row in self.rows if row.get("processing_status") == "INVALID"
        ]


def iter_n_dirs(
    shot_dir: Path, n_min: int = 1, n_max: int = 10
) -> list[tuple[int, Path]]:
    """Return existing requested ``N#`` directories in numeric order."""
    return [
        (n, shot_dir / f"N{n}")
        for n in range(n_min, n_max + 1)
        if (shot_dir / f"N{n}").is_dir()
    ]


def preflight_n_dirs(
    shot_dir: Path,
    *,
    n_min: int,
    n_max: int,
    pattern: str,
) -> list[tuple[int, Path, list[Path]]]:
    """Discover populated directories and abort before processing on missing datcon."""
    populated: list[tuple[int, Path, list[Path]]] = []
    for n, n_dir in iter_n_dirs(shot_dir, n_min=n_min, n_max=n_max):
        files = sorted(n_dir.glob(pattern))
        if not files:
            continue
        datcon_path = n_dir / f"datcon{n}"
        if not datcon_path.is_file():
            raise SystemExit(
                "Cannot process shot: required continuum file is missing for "
                f"N{n}: {datcon_path}"
            )
        try:
            with datcon_path.open("rb"):
                pass
        except OSError as exc:
            raise SystemExit(
                "Cannot process shot: required continuum file is unreadable for "
                f"N{n}: {datcon_path} ({type(exc).__name__}: {exc})"
            ) from exc
        populated.append((n, n_dir, files))
    return populated


def classify_gap_region(
    signed_delta: float,
    fraction_below_upper2: float,
    *,
    fraction_tae_threshold: float = DEFAULT_FRACTION_TAE_THRESHOLD,
    fraction_eae_threshold: float = DEFAULT_FRACTION_EAE_THRESHOLD,
    signed_delta_eae_threshold: float = DEFAULT_SIGNED_DELTA_EAE_THRESHOLD,
) -> str:
    """Apply the canonical ``sort_shot_mixed.py`` TAE/EAE/mixed rule."""
    if fraction_below_upper2 > fraction_tae_threshold:
        return "tae_like"
    if (
        fraction_below_upper2 < fraction_eae_threshold
        and signed_delta < signed_delta_eae_threshold
    ):
        return "eae_like"
    return "mixed"


def _base_row(path: Path, shot: str, n: int) -> dict[str, Any]:
    row = empty_rule_row()
    row.update(
        {
            "path": str(path.resolve()),
            "mode_key": portable_mode_key(path.resolve()),
            "shot": shot,
            "n": int(n),
            "rule_triggered_rules": stable_json([]),
            "rule_features": stable_json({}),
            "selected_final": "",
        }
    )
    return row


def _mark_invalid(
    row: dict[str, Any], reason: str, diagnostic_message: str
) -> dict[str, Any]:
    row.update(
        {
            "processing_status": "INVALID",
            "preprocessing_primary_reason": reason,
            "rule_decision": "INVALID",
            "rule_primary_reason": "",
            "rule_triggered_rules": stable_json([reason]),
            "final_decision": "INVALID",
            "decision_source": "preprocessing",
            "diagnostic_message": diagnostic_message,
        }
    )
    return row


def _inspect_mode_file(
    path: Path, *, expected_n: int
) -> tuple[dict[str, Any] | None, str, str]:
    try:
        mode, omega, gamma_d, ntor = load_mode_from_nova(str(path))
    except Exception as exc:
        return None, "MODE_LOAD_FAILED", f"{type(exc).__name__}: {exc}"

    mode = np.asarray(mode)
    if mode.ndim != 2 or mode.shape[0] <= 0 or mode.shape[1] <= 0:
        return None, "INVALID_METADATA", f"Unexpected mode shape {mode.shape}"

    nhar, nr = mode.shape
    metadata_values = [omega, gamma_d, ntor, nr, nhar]
    if not all(np.isfinite(value) for value in metadata_values):
        return (
            None,
            "INVALID_METADATA",
            "omega/gamma_d/ntor/nr/nhar contains a non-finite value",
        )
    if int(ntor) != int(expected_n):
        return (
            None,
            "NTOR_N_MISMATCH",
            f"Directory N{expected_n} disagrees with file metadata ntor={ntor}",
        )
    if nhar < 4 * int(ntor):
        return (
            None,
            "TOO_SMALL_NHAR",
            f"nhar={nhar} is smaller than required minimum 4*ntor={4 * int(ntor)}",
        )
    if not np.all(np.isfinite(mode)):
        return None, "NONFINITE_MODE_DATA", "Mode array contains non-finite values"

    weight_sum = float(np.sum(np.abs(mode) ** 2))
    if not np.isfinite(weight_sum) or weight_sum <= 0.0:
        return (
            None,
            "INVALID_WEIGHTS",
            f"Invalid amplitude-squared weight sum: {weight_sum}",
        )

    return (
        {
            "mode": mode,
            "omega": float(omega),
            "gamma_d": float(gamma_d),
            "ntor": int(ntor),
            "nr": int(nr),
            "nhar": int(nhar),
        },
        "",
        "",
    )


def _load_gap_scalars(
    path: Path, *, mode: np.ndarray, omega: float
) -> tuple[dict[str, float] | None, str, str]:
    try:
        _low2, upper2, *_ = load_datcon_for_mode(str(path), n_r=mode.shape[1])
    except Exception as exc:
        return None, "INVALID_CONTINUUM", f"{type(exc).__name__}: {exc}"
    if not np.any(np.isfinite(upper2)):
        return None, "INVALID_CONTINUUM", "upper2 has no finite values"
    try:
        scalars = upper2_scalars(mode, omega, upper2)
    except Exception as exc:
        return None, "INVALID_UPPER2_SCALARS", f"{type(exc).__name__}: {exc}"
    return scalars, "", ""


def preprocess_shot(
    shot_dir: str | Path,
    *,
    out_dir: str | Path | None = None,
    n_min: int = 1,
    n_max: int = 10,
    pattern: str = "egn*",
    fraction_tae_threshold: float = DEFAULT_FRACTION_TAE_THRESHOLD,
    fraction_eae_threshold: float = DEFAULT_FRACTION_EAE_THRESHOLD,
    signed_delta_eae_threshold: float = DEFAULT_SIGNED_DELTA_EAE_THRESHOLD,
) -> PreprocessResult:
    """Validate, fingerprint, and frequency-route all discovered shot modes."""
    source = Path(shot_dir).expanduser().resolve()
    if not source.is_dir():
        raise SystemExit(f"Shot directory not found: {source}")
    if n_min > n_max:
        raise ValueError("n_min must be less than or equal to n_max")
    if not 0.0 <= fraction_eae_threshold <= fraction_tae_threshold <= 1.0:
        raise ValueError(
            "frequency split thresholds must satisfy 0 <= fraction_eae_threshold "
            "<= fraction_tae_threshold <= 1"
        )
    if not math.isfinite(signed_delta_eae_threshold):
        raise ValueError("signed_delta_eae_threshold must be finite")

    populated = preflight_n_dirs(
        source, n_min=n_min, n_max=n_max, pattern=pattern
    )
    rows: list[dict[str, Any]] = []
    for n, _n_dir, files in populated:
        for path in files:
            row = _base_row(path, source.name, n)
            datcon_path = datcon_path_for_mode(path, n)
            try:
                row["input_fingerprint"] = input_fingerprint(path, datcon_path)
            except Exception as exc:
                rows.append(
                    _mark_invalid(
                        row,
                        "INPUT_FINGERPRINT_FAILED",
                        f"{type(exc).__name__}: {exc}",
                    )
                )
                continue

            bundle, reason, message = _inspect_mode_file(path, expected_n=n)
            if bundle is None:
                rows.append(_mark_invalid(row, reason, message))
                continue

            row.update(
                {
                    "ntor": bundle["ntor"],
                    "nr": bundle["nr"],
                    "nhar": bundle["nhar"],
                    "omega": bundle["omega"],
                    "gamma_d": bundle["gamma_d"],
                }
            )
            radial_grid = np.linspace(0.0, 1.0, bundle["nr"])
            centroid = float(radial_centroid(bundle["mode"], radial_grid))
            width = float(radial_width(bundle["mode"], radial_grid, centroid))
            if not (math.isfinite(centroid) and math.isfinite(width)):
                rows.append(
                    _mark_invalid(
                        row,
                        "INVALID_RADIAL_FEATURES",
                        "radial centroid or width is non-finite",
                    )
                )
                continue
            row.update({"rad_loc": centroid, "rad_width": width})

            scalars, reason, message = _load_gap_scalars(
                path, mode=bundle["mode"], omega=bundle["omega"]
            )
            if scalars is None:
                rows.append(_mark_invalid(row, reason, message))
                continue
            gap_region = classify_gap_region(
                scalars["signed_delta"],
                scalars["fraction_below_upper2"],
                fraction_tae_threshold=fraction_tae_threshold,
                fraction_eae_threshold=fraction_eae_threshold,
                signed_delta_eae_threshold=signed_delta_eae_threshold,
            )
            row.update(
                {
                    "signed_delta": scalars["signed_delta"],
                    "fraction_below_upper2": scalars["fraction_below_upper2"],
                    "gap_region": gap_region,
                    "processing_status": (
                        "ROUTED_EAE" if gap_region == "eae_like" else "READY_FOR_RULES"
                    ),
                    "decision_source": (
                        "frequency_routing" if gap_region == "eae_like" else ""
                    ),
                }
            )
            rows.append(row)

    result = PreprocessResult(
        shot_dir=source,
        shot=source.name,
        rows=tuple(sorted(rows, key=rule_row_sort_key)),
    )
    if out_dir is not None:
        write_preprocess_outputs(result, Path(out_dir).expanduser())
    return result


def write_preprocess_outputs(result: PreprocessResult, out_dir: Path) -> None:
    """Write the three preprocessing-compatible output lists."""
    write_dict_csv(out_dir / "tae_like_all.csv", RULE_OUTPUT_FIELDS, result.tae_rows)
    write_dict_csv(out_dir / "eae_like.csv", RULE_OUTPUT_FIELDS, result.eae_rows)
    write_dict_csv(
        out_dir / "rejected_modes.csv", RULE_OUTPUT_FIELDS, result.invalid_rows
    )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Validate and split one NOVA shot into TAE-side, EAE-side, and "
            "invalid mode lists without running RF or CNN classification."
        )
    )
    parser.add_argument("--shot_dir", required=True, help="Shot containing N1, N2, ...")
    parser.add_argument("--out_dir", required=True, help="Output directory")
    parser.add_argument("--n_min", type=int, default=1, help="Smallest N# to scan")
    parser.add_argument("--n_max", type=int, default=10, help="Largest N# to scan")
    parser.add_argument("--pattern", default="egn*", help="Mode-file glob")
    parser.add_argument(
        "--fraction_tae_threshold", type=float, default=DEFAULT_FRACTION_TAE_THRESHOLD
    )
    parser.add_argument(
        "--fraction_eae_threshold", type=float, default=DEFAULT_FRACTION_EAE_THRESHOLD
    )
    parser.add_argument(
        "--signed_delta_eae_threshold",
        type=float,
        default=DEFAULT_SIGNED_DELTA_EAE_THRESHOLD,
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    result = preprocess_shot(
        args.shot_dir,
        out_dir=args.out_dir,
        n_min=args.n_min,
        n_max=args.n_max,
        pattern=args.pattern,
        fraction_tae_threshold=args.fraction_tae_threshold,
        fraction_eae_threshold=args.fraction_eae_threshold,
        signed_delta_eae_threshold=args.signed_delta_eae_threshold,
    )
    print(f"Shot: {result.shot}")
    print(f"Discovered modes: {len(result.rows)}")
    print(f"TAE-side modes: {len(result.tae_rows)}")
    print(f"EAE-like modes: {len(result.eae_rows)}")
    print(f"Invalid modes: {len(result.invalid_rows)}")
    print(f"Wrote outputs to: {Path(args.out_dir).expanduser()}")


if __name__ == "__main__":
    main()
