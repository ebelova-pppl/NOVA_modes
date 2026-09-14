"""Compare local median backgrounds for individual signed edge peaks.

python audits/edge_secondary_peaks_20260913/local_background.py \
  --baseline-dir outputs/review_edge_secondary_peaks_20260913 \
  --out-dir outputs/review_edge_local_background_20260913
Read-only decision projection; no production sorting or label changes.
"""
import argparse
from collections import Counter
from concurrent.futures import ProcessPoolExecutor
import json
from pathlib import Path

import numpy as np

from signed_harmonics import (
    REPO, read, write, load_mode_from_nova, input_fingerprint,
    datcon_path_for_mode, sha256_file, _signed_local_extrema,
    _signed_halfmax_component,
)

PEAK_R_MIN = 0.95
AMPLITUDE_MIN = 0.7
CONTRAST_MIN = 3.0
WINDOW_WIDTHS = (0.05, 0.10)
LOBE_WIDTHS = (1.5, 2.0)


def edge_sample(index, nr):
    return index >= PEAK_R_MIN * (nr - 1) - 1e-12


def background_samples(r, peak_r, window_width, left, right, exclude):
    # Intersect the centered window with the native domain; never extrapolate
    # or shift it inward. Exclusion removes samples, not an estimated shape.
    mask = np.abs(r - peak_r) <= window_width / 2 + 1e-12
    if exclude:
        mask &= ~((r >= left - 1e-12) & (r <= right + 1e-12))
    return mask


def measure(source):
    path = Path(source["path"])
    fingerprint = input_fingerprint(path, datcon_path_for_mode(path))
    assert fingerprint == source["input_fingerprint"], path
    mode, _, _, _ = load_mode_from_nova(str(path))
    assert np.isfinite(mode).all()
    assert np.isclose(np.max(abs(mode)), 1, rtol=1e-10, atol=1e-12)
    nr = mode.shape[1]
    r = np.arange(nr, dtype=float) / (nr - 1)
    b = np.max(abs(mode), axis=0)
    edge = np.array([edge_sample(i, nr) for i in range(nr)])
    peaks = []
    for m, profile in enumerate(mode):
        if np.max(abs(profile[edge])) < AMPLITUDE_MIN:
            continue
        for i, _, _ in _signed_local_extrema(profile):
            amplitude = float(abs(profile[i]))
            if not edge_sample(i, nr) or amplitude < AMPLITUDE_MIN:
                continue
            left, right, width_r, width_grid, touches = _signed_halfmax_component(
                profile, peak_index=i, radial_grid=r)
            if width_grid > max(LOBE_WIDTHS) + 1e-12:
                continue
            for window in WINDOW_WIDTHS:
                for exclude in (False, True):
                    mask = background_samples(r, r[i], window, left, right, exclude)
                    n = int(mask.sum())
                    median = float(np.median(b[mask])) if n else None
                    mean = float(np.mean(b[mask])) if n else None
                    # Report undefined/zero backgrounds explicitly, without
                    # inventing a contrast or using them to trigger the gate.
                    contrast = amplitude / median if median is not None and median > 0 else None
                    peaks.append(dict(stored_index=m, peak_index=i, r_peak=float(r[i]),
                        signed_amplitude=float(profile[i]), amplitude=amplitude,
                        width_grid=width_grid, width_r=width_r, inner_edge=left,
                        outer_edge=right, touches_boundary=touches,
                        window_width=window, exclude_tested_lobe=exclude,
                        window_left=max(0., float(r[i]) - window / 2),
                        window_right=min(1., float(r[i]) + window / 2),
                        background_sample_count=n, background_median=median,
                        background_mean=mean, contrast=contrast))
    assert input_fingerprint(path, datcon_path_for_mode(path)) == fingerprint
    result = {k: source[k] for k in ("path", "mode_key", "cohort", "training_label", "input_fingerprint")}
    result.update(latest12=source["latest12"] == "True", nr=nr, peaks=peaks)
    return result


def selected_peak(row, window, exclude, width):
    peaks = [p for p in row["peaks"] if p["window_width"] == window
             and p["exclude_tested_lobe"] == exclude
             and p["width_grid"] <= width + 1e-12
             and p["contrast"] is not None and p["contrast"] >= CONTRAST_MIN]
    return min(peaks, key=lambda p: (p["width_grid"], -p["amplitude"], p["stored_index"], p["peak_index"])) if peaks else None


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--baseline-dir", type=Path, required=True, help="Previous audit's native input manifest and verification receipt")
    parser.add_argument("--out-dir", type=Path, required=True, help="Empty directory for audit measurements and projections")
    args = parser.parse_args()
    args.out_dir.mkdir(parents=True, exist_ok=True)
    assert not any(args.out_dir.iterdir()), "Choose an empty output directory"
    manifest = args.baseline_dir / "measurements.csv"
    receipt = args.baseline_dir / "summary.json"
    baseline = json.loads(receipt.read_text())
    source_hashes = dict(baseline["source_sha256"])
    source_hashes.update({str(p): sha256_file(p) for p in (
        Path(__file__), Path(__file__).with_name("signed_harmonics.py"), manifest, receipt)})
    saved = baseline["saved_output_sha256"]
    for path, digest in {**source_hashes, **saved}.items():
        assert sha256_file(Path(path)) == digest, path
    sources = read(manifest)
    assert len(sources) == baseline["measured"]
    assert len({s["mode_key"] for s in sources}) == len(sources)
    rows = []
    with ProcessPoolExecutor(max_workers=4) as pool:
        for i, result in enumerate(pool.map(measure, sources), 1):
            rows.append(result)
            if i % 500 == 0:
                print(f"Measured {i}/{len(sources)} native arrays", flush=True)
    (args.out_dir / "measurements.json").write_text(json.dumps(rows, allow_nan=False) + '\n')
    comparisons, review = [], []
    for window in WINDOW_WIDTHS:
        for exclude in (False, True):
            for width in LOBE_WIDTHS:
                chosen = [(row, selected_peak(row, window, exclude, width)) for row in rows]
                chosen = [(row, p) for row, p in chosen if p is not None]
                comparisons.append(dict(window_width=window, exclude_tested_lobe=exclude, width_max_grid=width,
                    training_good=sum(row["cohort"] == "training" and row["training_label"] == "good" for row, p in chosen),
                    training_bad=sum(row["cohort"] == "training" and row["training_label"] == "bad" for row, p in chosen),
                    pilot39=sum(row["cohort"] == "pilot" for row, p in chosen),
                    latest12=sum(row["cohort"] == "pilot" and row["latest12"] for row, p in chosen),
                    pending_envelope=sum(row["cohort"] == "pilot_pending_envelope" for row, p in chosen)))
                for row, p in chosen:
                    review.append(dict({k: v for k, v in row.items() if k != "peaks"}, **p, width_max_grid=width))
    write(args.out_dir / "comparison.csv", comparisons, list(comparisons[0]))
    if review:
        write(args.out_dir / "review_candidates.csv", review, list(review[0]))
    for path, digest in {**source_hashes, **saved}.items():
        assert sha256_file(Path(path)) == digest, path
    summary = dict(status="AUDIT_ONLY", peak_r_min_inclusive=PEAK_R_MIN,
        amplitude_min=AMPLITUDE_MIN, contrast_min=CONTRAST_MIN,
        background="median(max_h abs(xi_h)) in centered window intersected with native domain",
        exclusion="optional removal of samples within this tested signed lobe's full FWHM interval; other peaks remain",
        measured=len(rows), cohort_counts=dict(Counter(row["cohort"] for row in rows)),
        n_radial=sorted({row["nr"] for row in rows}),
        unavailable_backgrounds=sum(p["contrast"] is None for row in rows for p in row["peaks"]),
        comparisons=comparisons, source_sha256=source_hashes, saved_output_sha256=saved)
    (args.out_dir / "summary.json").write_text(json.dumps(summary, indent=2, allow_nan=False) + '\n')
    print(json.dumps({k: v for k, v in summary.items() if not k.endswith("sha256")}, indent=2))


if __name__ == "__main__":
    main()
