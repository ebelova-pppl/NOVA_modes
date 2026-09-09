"""Non-blind audit of near-axis amplitude and radial energy concentration.

python audits/axis_amplitude_20260909/audit.py --rules-root /path/to/rules \
  --training-root /path/to/training --out-dir outputs/review_axis_amplitude_20260909
This measures candidate limits; it does not change production or truth labels.
"""

import argparse
from concurrent.futures import ProcessPoolExecutor
import csv
import json
from pathlib import Path
import sys

import numpy as np

HERE = Path(__file__).resolve().parent
REPO = HERE.parents[1]
sys.path[:0] = [str(REPO / "src"), str(REPO / "scripts")]
from make_tae_like_list import _inspect_mode_file, _load_gap_data
from cont_features import _energy_fraction_in_window
from tae_eae_features import classify_gap_region
from tae_rule_engine import evaluate_mode, ContinuumCrossingConfig
from tae_rule_io import (
    input_fingerprint,
    datcon_path_for_mode,
    sha256_file,
    stable_json,
)

TARGET = "nstxuG142301L94/N5/egn05w.2135E+02"
RADII = (0.01, 0.015)
LIMITS = (0.5, 0.7, 0.8)
ENERGY_RADII = (0.01, 0.015, 0.02, 0.03, 0.05, 0.1)
COMBINED_AMPLITUDE_RADIUS = 0.015
COMBINED_AMPLITUDE_LIMIT = 0.5
COMBINED_ENERGY_RADIUS = 0.05
COMBINED_ENERGY_MIN = 0.5


def read(path):
    with path.open() as f:
        return list(csv.DictReader(f))


def write(path, rows, fields=None):
    with path.open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fields or list(rows[0]), lineterminator="\n")
        w.writeheader()
        w.writerows(rows)


def evaluate(path, bundle, fingerprint):
    gap, reason, message = _load_gap_data(
        path, mode=bundle["mode"], omega=bundle["omega"]
    )
    assert gap is not None, (path, reason, message)
    region = classify_gap_region(**gap.scalars)
    assert region in {"tae_like", "mixed"}, (path, region)
    row = dict(
        path=str(path),
        mode_key="/".join(path.parts[-3:]),
        shot=path.parents[1].name,
        ntor=bundle["ntor"],
        omega=bundle["omega"],
        gamma_d=bundle["gamma_d"],
        gap_region=region,
        input_fingerprint=fingerprint,
    )
    result = evaluate_mode(
        row,
        mode=bundle["mode"],
        low2=gap.low2,
        high2=gap.high2,
        continuum_crossing_config=ContinuumCrossingConfig(w_cross_threshold=None),
    )
    assert result.decision in {"REVIEW", "BAD"}, result.diagnostic_message
    return result, gap


def measure(task):
    cohort, supplied, root = task
    path = (
        root / supplied["path"] if cohort == "training_good" else Path(supplied["path"])
    )
    key = "/".join(path.parts[-3:])
    bundle, reason, message = _inspect_mode_file(
        path, expected_n=int(path.parent.name[1:])
    )
    assert bundle is not None, (key, reason, message)
    assert bundle["nr"] == 201, key
    fingerprint = input_fingerprint(path, datcon_path_for_mode(path))
    if cohort == "shot_good":
        assert fingerprint == supplied["input_fingerprint"], key
        assert supplied["rule_decision"] == "REVIEW" and not supplied["manual_decision"]
    mode = bundle["mode"]
    r = np.linspace(0, 1, mode.shape[1])
    row = dict(
        path=str(path),
        mode_key=key,
        cohort=cohort,
        input_fingerprint=fingerprint,
        nr=bundle["nr"],
        baseline_decision="GOOD" if cohort == "shot_good" else "NOT_NEEDED",
        baseline_reason=supplied.get("rule_primary_reason", ""),
    )
    for radius in RADII:
        indices = np.flatnonzero(r <= radius)
        window = np.abs(mode[:, indices])
        h, j = np.unravel_index(np.argmax(window), window.shape)
        row[f"A_r{radius}"] = float(window[h, j])
        row[f"h_r{radius}"] = int(h)
        row[f"peak_r_r{radius}"] = float(r[indices[j]])
    radial_energy = np.sum(mode**2, axis=0)
    for radius in ENERGY_RADII:
        row[f"F_r{radius}"] = _energy_fraction_in_window(
            radial_energy, r, radius / 2, radius / 2
        )
    row["combined_candidate"] = bool(
        row[f"A_r{COMBINED_AMPLITUDE_RADIUS}"] > COMBINED_AMPLITUDE_LIMIT
        and row[f"F_r{COMBINED_ENERGY_RADIUS}"] > COMBINED_ENERGY_MIN
    )
    if cohort == "training_good" and row["A_r0.015"] > min(LIMITS):
        result, _ = evaluate(path, bundle, fingerprint)
        row["baseline_decision"] = "GOOD" if result.decision == "REVIEW" else "BAD"
        row["baseline_reason"] = result.primary_reason
    for radius in RADII:
        for limit in LIMITS:
            row[f"cap_r{radius}_A{limit}"] = row[f"A_r{radius}"] > limit
    return row


def target_diagnostic(supplied, out_dir):
    import matplotlib.pyplot as plt

    path = Path(supplied["path"])
    bundle, _, _ = _inspect_mode_file(path, expected_n=5)
    fingerprint = input_fingerprint(path, datcon_path_for_mode(path))
    assert fingerprint == supplied["input_fingerprint"]
    result, gap = evaluate(path, bundle, fingerprint)
    assert result.decision == supplied["rule_decision"] == "REVIEW"
    assert json.loads(stable_json(result.features)) == json.loads(
        supplied["rule_features"]
    )
    mode = bundle["mode"]
    r = np.linspace(0, 1, mode.shape[1])
    h = int(np.unravel_index(np.argmax(np.abs(mode)), mode.shape)[0])
    W = np.sum(mode**2, axis=0)
    order = np.argsort(np.max(np.abs(mode), axis=1))[::-1][:8]
    fig, axes = plt.subplots(2, 2, figsize=(12, 8), constrained_layout=True)
    for i in order:
        axes[0, 0].plot(r, mode[i], label=f"h={i}")
        axes[0, 1].plot(r, mode[i], ".-", label=f"h={i}")
    axes[0, 0].set(title="Signed harmonics, full radius", ylabel="Normalized amplitude")
    axes[0, 1].set(title="Near-axis samples (stored harmonic indices)", xlim=(0, 0.06))
    axes[0, 1].axvspan(0, 0.01, alpha=0.1, color="tab:red")
    axes[0, 1].legend(fontsize=8, ncol=2)
    axes[1, 0].plot(r, W / W.max(), ".-")
    axes[1, 0].set(
        title="Total radial energy near axis", xlim=(0, 0.15), ylabel="W / max(W)"
    )
    axes[1, 1].plot(r, np.sqrt(gap.low2), label="Lower")
    axes[1, 1].plot(r, np.sqrt(gap.high2), label="Upper")
    axes[1, 1].axhline(bundle["omega"], color="k", ls="--", label="Mode")
    axes[1, 1].set(
        title="TAE continuum",
        ylabel="Absolute frequency",
        ylim=(0, 2 * bundle["omega"]),
    )
    axes[1, 1].legend()
    for ax in axes.flat:
        ax.set_xlabel("Normalized radius r")
        ax.grid(alpha=0.2)
    fig.suptitle(TARGET + " — non-blind diagnosis")
    fig.savefig(out_dir / "L94_N5_2135.png", dpi=160)
    plt.close(fig)
    return dict(
        mode_key=TARGET,
        input_fingerprint=fingerprint,
        prior_seen=True,
        raw_v20_features_and_decision_reproduced=True,
        dominant_stored_harmonic=h,
        samples=[
            dict(r=float(r[j]), signed_amplitude=float(mode[h, j])) for j in range(7)
        ],
        axis_features=result.features["boundary_features"]["axis_artifact"],
        energy_features=result.features["resolution_features"][
            "interior_unresolved_envelope"
        ],
    )


def main():
    p = argparse.ArgumentParser(description=__doc__)
    for option in ("rules-root", "training-root", "out-dir"):
        p.add_argument("--" + option, type=Path, required=True)
    args = p.parse_args()
    args.out_dir.mkdir(parents=True, exist_ok=True)
    batch = read(HERE.parent / "extremum_floor_20260909/clearance_shot_summary.csv")
    shots = []
    for shot in batch:
        shots.extend(
            r
            for r in read(args.rules_root / shot["shot"] / "all_modes_rules.csv")
            if r["final_decision"] == "GOOD"
        )
    training = [
        r
        for r in read(REPO / "training_labels/tae_like_train.csv")
        if r["validity"] == "good"
    ]
    assert len(shots) == 950 and len(training) == 575
    target = next(r for r in shots if r["mode_key"] == TARGET)
    diagnosis = target_diagnostic(target, args.out_dir)
    tasks = [("shot_good", r, args.rules_root) for r in shots]
    tasks.extend(("training_good", r, args.training_root) for r in training)
    with ProcessPoolExecutor(max_workers=4) as pool:
        rows = list(pool.map(measure, tasks, chunksize=8))
    write(args.out_dir / "all_measurements.csv", rows)
    flagged = [
        r
        for r in rows
        if any(r[f"cap_r{radius}_A{limit}"] for radius in RADII for limit in LIMITS)
    ]
    write(HERE / "flagged_modes.csv", flagged, fields=list(rows[0]))
    combined = [r for r in rows if r["combined_candidate"]]
    write(HERE / "combined_changes.csv", combined, fields=list(rows[0]))
    comparisons = []
    for radius in RADII:
        for limit in LIMITS:
            key = f"cap_r{radius}_A{limit}"
            comparisons.append(
                dict(
                    radius=radius,
                    amplitude_limit=limit,
                    newly_rejected_shot_good=sum(
                        r[key] and r["cohort"] == "shot_good" for r in rows
                    ),
                    training_good_flagged=sum(
                        r[key] and r["cohort"] == "training_good" for r in rows
                    ),
                    newly_rejected_training_good=sum(
                        r[key]
                        and r["cohort"] == "training_good"
                        and r["baseline_decision"] == "GOOD"
                        for r in rows
                    ),
                )
            )
    energy_comparisons = []
    for radius in (0.03, 0.05, 0.1):
        for fraction in (0.2, 0.3, 0.5):
            selected = [
                r
                for r in rows
                if r[f"A_r{COMBINED_AMPLITUDE_RADIUS}"] > COMBINED_AMPLITUDE_LIMIT
                and r[f"F_r{radius}"] > fraction
            ]
            energy_comparisons.append(
                dict(
                    energy_radius=radius,
                    energy_fraction_min=fraction,
                    newly_rejected_shot_good=sum(
                        r["cohort"] == "shot_good" for r in selected
                    ),
                    training_good_flagged=sum(
                        r["cohort"] == "training_good" for r in selected
                    ),
                )
            )
    assert len(combined) == 1 and combined[0]["mode_key"] == TARGET
    summary = dict(
        status="audit_only_not_adopted",
        prior_seen=True,
        shot_survivors=950,
        training_good_labels=575,
        comparisons=comparisons,
        combined_proposal=dict(
            amplitude_radius=COMBINED_AMPLITUDE_RADIUS,
            amplitude_min_exclusive=COMBINED_AMPLITUDE_LIMIT,
            energy_radius=COMBINED_ENERGY_RADIUS,
            energy_fraction_min_exclusive=COMBINED_ENERGY_MIN,
            newly_rejected_shot_good=1,
            newly_rejected_training_good=0,
            energy_definition="integral_0^R sum_h abs(xi_h)^2 dr / integral_0^1 sum_h abs(xi_h)^2 dr",
        ),
        energy_sensitivity=energy_comparisons,
        source_sha256={
            p: sha256_file(REPO / p)
            for p in (
                "scripts/tae_rule_engine.py",
                "scripts/tae_rule_config.py",
                "src/cont_features.py",
                "configs/rules/tae_rules_production_v8.yaml",
                "training_labels/tae_like_train.csv",
                "audits/axis_amplitude_20260909/audit.py",
            )
        },
    )
    (HERE / "target_diagnostic.json").write_text(json.dumps(diagnosis, indent=2) + "\n")
    (HERE / "summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    print(json.dumps(comparisons, indent=2), flush=True)
    print(json.dumps(summary["combined_proposal"], indent=2), flush=True)


if __name__ == "__main__":
    main()
