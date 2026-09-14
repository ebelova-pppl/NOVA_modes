"""Audit A>=0.7 and P_edge>=3 on individual signed edge lobes.

python audits/edge_secondary_peaks_20260913/signed_harmonics.py \
  --baseline-dir outputs/review_edge_secondary_peaks_20260913 \
  --out-dir outputs/review_edge_signed_peaks_20260913
The bounded baseline manifest supplies the existing raw input paths.
"""
import argparse
from collections import Counter
from concurrent.futures import ProcessPoolExecutor
import csv
import json
from pathlib import Path
import sys

import numpy as np

REPO = Path(__file__).resolve().parents[2]
sys.path[:0] = [str(REPO / "src"), str(REPO / "scripts")]
from nova_mode_loader import load_mode_from_nova
from tae_rule_engine import _signed_local_extrema, _signed_halfmax_component
from tae_rule_io import input_fingerprint, datcon_path_for_mode, sha256_file

RADII = (0.9, 0.925, 0.95, 0.97)
WIDTHS = (1.0, 1.25, 1.5, 2.0, 3.0, 4.0)
AMPLITUDE_MIN = 0.7
CONTRAST_MIN = 3.0


def read(path):
    with Path(path).open(newline="") as f:
        return list(csv.DictReader(f))


def write(path, rows, fields):
    with Path(path).open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fields, lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)


def beyond_radius(index, nr, cutoff):
    # Compare in native index space: a rounded r=0.9500000000000001 sample
    # must not accidentally satisfy the strict r>0.95 condition.
    return index > cutoff * (nr - 1) + 1e-12


def measure(source):
    path = Path(source["path"])
    fp = input_fingerprint(path, datcon_path_for_mode(path))
    assert fp == source["input_fingerprint"], path
    A, _, _, _ = load_mode_from_nova(str(path))
    assert np.isfinite(A).all()
    maximum = float(np.max(abs(A)))
    assert np.isclose(maximum, 1, rtol=1e-10, atol=1e-12), (path, maximum)
    nr = A.shape[1]
    r = np.arange(nr, dtype=float) / (nr - 1)
    B = np.max(abs(A), axis=0)
    background = float(np.median(B[r >= 0.9]))
    assert np.isclose(background, float(source["background_amplitude"]), rtol=1e-10, atol=1e-12)
    candidates = []
    if background > 0:
        edge_indices = np.array([i for i in range(nr) if beyond_radius(i, nr, min(RADII))])
        for m, profile in enumerate(A):
            if np.max(abs(profile[edge_indices])) < AMPLITUDE_MIN:
                continue
            for i, _, _ in _signed_local_extrema(profile):
                amplitude = float(abs(profile[i]))
                if not beyond_radius(i, nr, min(RADII)) or amplitude < AMPLITUDE_MIN:
                    continue
                contrast = amplitude / background
                if contrast < CONTRAST_MIN:
                    continue
                left, right, width_r, width_grid, touches = _signed_halfmax_component(
                    profile, peak_index=i, radial_grid=r)
                candidates.append(dict(stored_index=m, peak_index=i, r_peak=float(r[i]),
                                       signed_amplitude=float(profile[i]), amplitude=amplitude,
                                       contrast=contrast, width_grid=width_grid, width_r=width_r,
                                       inner_edge=left, outer_edge=right, touches_boundary=touches))
    assert input_fingerprint(path, datcon_path_for_mode(path)) == fp
    result = {key:source[key] for key in ("path", "mode_key", "cohort", "training_label", "input_fingerprint")}
    result.update(latest12=source["latest12"] == "True", nr=nr, maximum_amplitude=maximum,
                  background_amplitude=background, qualifying_amplitude_contrast_peaks=candidates)
    return result


def selected_peak(row, radius, width):
    peaks = [p for p in row["qualifying_amplitude_contrast_peaks"]
             if beyond_radius(p["peak_index"], row["nr"], radius) and p["width_grid"] <= width + 1e-12]
    return min(peaks, key=lambda p:(p["width_grid"], -p["amplitude"], p["stored_index"], p["peak_index"])) if peaks else None


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--baseline-dir", type=Path, required=True)
    parser.add_argument("--out-dir", type=Path, required=True)
    args = parser.parse_args()
    args.out_dir.mkdir(parents=True, exist_ok=True)
    assert not any(args.out_dir.iterdir()), "Choose an empty output directory"
    manifest = args.baseline_dir / "measurements.csv"
    receipt = args.baseline_dir / "summary.json"
    baseline = json.loads(receipt.read_text())
    for path, sha in baseline["source_sha256"].items():
        assert sha256_file(Path(path)) == sha, path
    saved = baseline["saved_output_sha256"]
    assert all(sha256_file(Path(p)) == sha for p, sha in saved.items())
    source = read(manifest)
    assert len(source) == baseline["measured"]
    assert len({r["mode_key"] for r in source}) == len(source)
    rows = []
    with ProcessPoolExecutor(max_workers=4) as pool:
        for i, result in enumerate(pool.map(measure, source), 1):
            rows.append(result)
            if i % 500 == 0:
                print(f"Measured {i}/{len(source)} native signed arrays", flush=True)
    (args.out_dir / "measurements.json").write_text(json.dumps(rows)+'\n')
    comparisons = []
    for radius in RADII:
        for width in WIDTHS:
            chosen = [(r, selected_peak(r, radius, width)) for r in rows]
            chosen = [(r,p) for r,p in chosen if p is not None]
            comparisons.append(dict(r_peak_strictly_greater_than=radius, width_max_grid=width,
                training_good=sum(r["cohort"] == "training" and r["training_label"] == "good" for r,p in chosen),
                training_bad=sum(r["cohort"] == "training" and r["training_label"] == "bad" for r,p in chosen),
                pilot39=sum(r["cohort"] == "pilot" for r,p in chosen),
                latest12=sum(r["cohort"] == "pilot" and r["latest12"] for r,p in chosen),
                pending_envelope=sum(r["cohort"] == "pilot_pending_envelope" for r,p in chosen)))
            if radius == 0.9 and width in (1.5, 2.0):
                fields = [k for k in rows[0] if k != "qualifying_amplitude_contrast_peaks"] + [
                    "stored_index", "peak_index", "r_peak", "signed_amplitude", "amplitude", "contrast",
                    "width_grid", "width_r", "inner_edge", "outer_edge", "touches_boundary"]
                review = [dict({k:v for k,v in row.items() if k != "qualifying_amplitude_contrast_peaks"}, **p)
                          for row,p in chosen]
                write(args.out_dir / f"new_rejections_width_{width}.csv", review, fields)
    write(args.out_dir / "comparison.csv", comparisons, list(comparisons[0]))
    assert all(sha256_file(Path(p)) == sha for p, sha in saved.items())
    summary = dict(status="AUDIT_ONLY", amplitude_min=AMPLITUDE_MIN, contrast_min=CONTRAST_MIN,
                   background="median(max_h abs(xi_h)) over r>=0.9; native samples; no spike removal",
                   amplitude_normalization="verified supplied max absolute amplitude is one; no renormalization",
                   radius_comparison="strict > in native index space", width_comparison="inclusive <=; signed-lobe FWHM",
                   measured=len(rows), n_radial=sorted({r["nr"] for r in rows}),
                   cohort_counts=dict(Counter(r["cohort"] for r in rows)), comparisons=comparisons,
                   source_sha256={str(p):sha256_file(p) for p in (Path(__file__),manifest,receipt,
                                  REPO / "scripts/tae_rule_engine.py", REPO / "src/nova_mode_loader.py")},
                   saved_output_sha256=saved)
    (args.out_dir / "summary.json").write_text(json.dumps(summary,indent=2)+'\n')
    print(json.dumps({k:v for k,v in summary.items() if not k.endswith('sha256')},indent=2))


if __name__ == "__main__":
    main()
