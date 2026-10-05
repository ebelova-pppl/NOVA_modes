"""Check current N1--N10 binary inputs in the six previously NaN-held shots.

Read only with respect to DiTw; write evidence into this audit and runtime dir.
Example: python audits/recalculated_input_check_20260930/check_nan.py \
    --data-root "$NOVA_DITW_ROOT" \
    --runtime-dir outputs/review_recalculated_input_check_20260930
"""
import argparse
import csv
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import sys

import numpy as np

HERE = Path(__file__).resolve().parent
REPO = HERE.parents[1]
sys.path.insert(0, str(REPO / "src"))
from nova_mode_loader import load_mode_from_nova
from cont_features import load_datcon_for_mode


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def write_csv(path, rows, fields):
    with path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-root", required=True, type=Path)
    parser.add_argument("--runtime-dir", required=True, type=Path)
    args = parser.parse_args()
    selection = list(csv.DictReader((HERE / "selection.csv").open()))
    shots = [r["shot"] for r in selection if r["check_nan"] == "True"]
    expected_n = dict(zip(
        ["nstxuG142301D46", "nstxuG142301M21", "nstxuE203653A02t017",
         "nstxuE203655F01t020", "nstxuE203655F01t030", "nstxuE205042A01t025"],
        [7, 4, 6, 6, 8, 10]))
    records, summary, sources, inventories = [], [], {}, {}
    for shot in shots:
        for n in range(1, 11):
            directory = args.data_root / shot / f"N{n}"
            paths = sorted(p for p in directory.glob("egn*") if p.is_file())
            inventories[str(directory)] = [p.name for p in paths]
            rows, grids = [], set()
            dc = directory / f"datcon{n}"
            if dc.exists():
                sources[str(dc)] = sha(dc)
            for path in paths:
                digest = sha(path)
                sources[str(path)] = digest
                row = dict(shot=shot, n=n, path=str(path.relative_to(args.data_root)),
                           sha256=digest, nr="", omega="", gamma_d="",
                           gamma_nan=False, raw_nonfinite=0, error="",
                           modified_utc=datetime.fromtimestamp(path.stat().st_mtime, timezone.utc).isoformat())
                try:
                    raw = np.fromfile(path, dtype=np.float64)
                    row["raw_nonfinite"] = int(np.count_nonzero(~np.isfinite(raw)))
                    mode, omega, gamma, ntor = load_mode_from_nova(path)
                    row.update(nr=mode.shape[1], omega=omega, gamma_d=gamma,
                               gamma_nan=bool(np.isnan(gamma)))
                    if ntor != n or not np.isfinite(omega) or omega <= 0:
                        raise ValueError("Invalid frequency/toroidal number")
                    if row["raw_nonfinite"]:
                        raise ValueError("Nonfinite raw values")
                    if mode.shape[1] not in grids:
                        load_datcon_for_mode(str(path), mode.shape[1])
                        grids.add(mode.shape[1])
                except (ValueError, OSError, IndexError, OverflowError, ZeroDivisionError) as exc:
                    row["error"] = str(exc)
                rows.append(row)
            records.extend(rows)
            result = dict(shot=shot, n=n, original_nan_scope=n == expected_n[shot],
                          current_modes=len(rows), gamma_nan=sum(r["gamma_nan"] for r in rows),
                          nonfinite_files=sum(r["raw_nonfinite"] > 0 for r in rows),
                          input_errors=sum(bool(r["error"]) for r in rows),
                          nr_values=",".join(str(g) for g in sorted(grids)))
            summary.append(result)
            if n == expected_n[shot] or result["input_errors"]:
                print(result, flush=True)
    # A changing inventory or payload invalidates this snapshot.
    for directory, names in inventories.items():
        assert sorted(p.name for p in Path(directory).glob("egn*") if p.is_file()) == names, directory
    for name, digest in sources.items():
        assert Path(name).is_file() and sha(Path(name)) == digest, name
    write_csv(HERE / "nan_summary.csv", summary, list(summary[0]))
    write_csv(HERE / "remaining_invalid_modes.csv", [r for r in records if r["error"]], list(records[0]))
    write_csv(args.runtime_dir / "nan_mode_inventory.csv", records, list(records[0]))
    metadata = dict(created_utc=datetime.now(timezone.utc).isoformat(),
                    data_root=str(args.data_root), shots=shots, source_sha256=sources,
                    inventories=inventories, script_sha256=sha(Path(__file__)),
                    mode_count=len(records), gamma_nan_count=sum(r["gamma_nan"] for r in records),
                    invalid_files=sum(bool(r["error"]) for r in records))
    (args.runtime_dir / "nan_metadata.json").write_text(json.dumps(metadata, indent=2) + "\n")
    print({k: metadata[k] for k in ("mode_count", "gamma_nan_count", "invalid_files")}, flush=True)


if __name__ == "__main__":
    main()
