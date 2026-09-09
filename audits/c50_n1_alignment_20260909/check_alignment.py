"""Compare C50 N1 crossings with the original stability-run singularity log.

Example:
  python audits/c50_n1_alignment_20260909/check_alignment.py \
    --shot-dir /path/to/nstxuG142301C50 --rules-dir /path/to/rules/SHOT \
    --out-dir outputs/review_c50_n1_alignment_20260909

This is a data-consistency diagnostic, not a classifier or continuum repair.
"""

import argparse
import csv
import json
from pathlib import Path
import re
import sys

REPO = Path(__file__).resolve().parents[2]
sys.path[:0] = [str(REPO / "src"), str(REPO / "scripts")]

import numpy as np
from cont_features import load_datcon_for_mode, continuum_crossing_records
from nova_mode_loader import load_mode_from_nova
from tae_rule_io import input_fingerprint, sha256_file


def log_records(path):
    text = path.read_text(errors="replace")
    records = []
    for match in re.finditer(r"hhh,om\s+([^\n]+)", text):
        start = text.rfind("Singularities are expected", 0, match.start())
        if start < 0:
            continue
        radii = [
            float(v)
            for v in re.findall(
                r"ixmax vs ising\s+\d+\s+\d+\s+(\S+)", text[start : match.start()]
            )
        ]
        if radii:
            records.append((float(match[1].split()[1]), radii))
    return records


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("shot-dir", "rules-dir", "out-dir"):
        parser.add_argument("--" + name, type=Path, required=True)
    args = parser.parse_args()
    args.out_dir.mkdir(parents=True, exist_ok=True)
    manifest = args.rules_dir / "all_modes_rules.csv"
    with manifest.open() as handle:
        shot_rows = list(csv.DictReader(handle))
    rows = [
        r
        for r in shot_rows
        if r["ntor"] == "1" and r["processing_status"] == "RULE_EVALUATED"
    ]
    assert len(rows) == 14
    n1 = args.shot_dir / "N1"
    sources = [manifest, n1 / "out_go", n1 / "datcon1", Path(__file__)]
    for n in range(1, 11):
        sources += [
            args.shot_dir / f"N{n}" / name
            for name in (
                "qprofile.txt",
                "bprofile.txt",
                "rgrid.dat",
                "gridparam",
                "mapdsk",
                "mpout1",
                "transp.dat",
            )
        ]
    hashes = {str(p): sha256_file(p) for p in sources}
    records = log_records(n1 / "out_go")
    measurements = []
    for row in rows:
        path = n1 / Path(row["path"]).name
        mode, omega, gamma, ntor = load_mode_from_nova(str(path))
        fingerprint = input_fingerprint(path, n1 / "datcon1")
        assert fingerprint == row["input_fingerprint"]
        assert ntor == 1 and mode.shape == (22, 201)
        low, high, i1, i2 = load_datcon_for_mode(str(path), 201)
        raw = np.loadtxt(n1 / "datcon1", skiprows=1)
        assert (i1, i2) == (3, 199)
        assert np.array_equal(low[i1 - 1 : i2], raw[:, 0])
        assert np.array_equal(high[i1 - 1 : i2], raw[:, 1])
        logged_omega2, singularities = min(records, key=lambda v: abs(v[0] - omega**2))
        assert np.isclose(logged_omega2, omega**2, rtol=1e-12, atol=0)
        crossings = continuum_crossing_records(mode, omega, low, high)
        assert crossings == json.loads(row["rule_features"])["crossing_records"]
        # The inner TAE-gap crossing is distinct from the crowded edge spectrum.
        core = [c for c in crossings if 0.03 <= c["r_cross"] < 0.75]
        assert len(core) == 1
        crossing = core[0]
        rc = crossing["r_cross"]
        nearest = min(singularities, key=lambda value: abs(value - rc))
        measurements.append(
            dict(
                mode_key=row["mode_key"],
                input_fingerprint=fingerprint,
                omega=omega,
                omega2=omega**2,
                logged_omega2=logged_omega2,
                boundary=crossing["boundary"],
                datcon_crossing_r=rc,
                nearest_logged_singularity_r=nearest,
                outward_offset_r=rc - nearest,
                outward_offset_grid=(rc - nearest) * 200,
                original_rules_decision=row["final_decision"],
                original_rules_reason=row["rule_primary_reason"],
            )
        )
    offsets = np.array([r["outward_offset_grid"] for r in measurements])
    assert np.all(offsets > 2)
    # These files establish current consistency, not the historic solver cache.
    equal_across_n = {
        name: len({hashes[str(args.shot_dir / f"N{n}" / name)] for n in range(1, 11)})
        == 1
        for name in (
            "qprofile.txt",
            "bprofile.txt",
            "rgrid.dat",
            "gridparam",
            "mapdsk",
            "mpout1",
            "transp.dat",
        )
    }
    assert all(equal_across_n.values())
    assert all(sha256_file(Path(p)) == h for p, h in hashes.items())
    with (args.out_dir / "measurements.csv").open("w", newline="") as handle:
        writer = csv.DictWriter(
            handle, fieldnames=list(measurements[0]), lineterminator="\n"
        )
        writer.writeheader()
        writer.writerows(measurements)
    summary = dict(
        shot=args.shot_dir.name,
        n1_tae_side_modes=len(rows),
        n1_total_modes=sum(r["ntor"] == "1" for r in shot_rows),
        min_offset_grid=float(offsets.min()),
        max_offset_grid=float(offsets.max()),
        median_offset_grid=float(np.median(offsets)),
        continuum_cleanup_changes=0,
        current_inputs_identical_across_n=equal_across_n,
        historic_cache_targets={
            name: str((n1 / name).readlink()) for name in ("equout", "equou1")
        },
        historic_cache_available={
            name: (n1 / name).exists() for name in ("equout", "equou1")
        },
        source_sha256=hashes,
    )
    (args.out_dir / "summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    print(
        json.dumps({k: v for k, v in summary.items() if k != "source_sha256"}, indent=2)
    )


if __name__ == "__main__":
    main()
