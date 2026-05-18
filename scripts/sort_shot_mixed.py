#!/usr/bin/env python3
"""
sort_shot_mixed.py
==================

Process one mixed TAE/EAE shot directory without moving files:

1. validate candidate NOVA mode files,
2. route valid modes into TAE-like / EAE-like / mixed groups,
3. run both the RF classifier and raw CNN on TAE-like modes,
4. combine RF/CNN probabilities with a confidence policy,
5. remove close-frequency near-duplicates from the GOOD TAE list, and
6. write shot-level CSV outputs, summaries, and optional QC plots.

The script intentionally reuses the shared helpers that define the current
project conventions rather than carrying a second copy of the scientific logic.

Output-list note:
- `flagged_tae_like.csv` is an overlapping QC list, not a separate class.
  It contains already-classified TAE-side modes that are borderline or show
  RF/CNN disagreement, so rows may also appear in `good_tae_unchecked.csv`
  or `bad_tae_like.csv`.
"""

from __future__ import annotations

import argparse
import csv
import math
from collections import defaultdict
from pathlib import Path
from typing import Any, Iterable, Sequence

import joblib
import numpy as np

from cnn_infer_common import load_cnn_classifier
from cont_features import load_datcon_for_mode
from nova_mode_loader import load_mode_from_nova
from sort_shot import (
    build_mode_dict,
    classify_mode_rf,
    iter_n_dirs,
    postprocess_good_modes,
    write_cluster_csv,
    write_cluster_report,
)
from tae_eae_features import upper2_scalars


ALL_OUTPUT_FIELDS = [
    "path",
    "shot",
    "n",
    "ntor",
    "nr",
    "nhar",
    "omega",
    "gamma_d",
    "status",
    "gap_region",
    "signed_delta",
    "fraction_below_upper2",
    "p_rf_good",
    "p_cnn_good",
    "p_avg",
    "final_label",
    "tier",
    "model_disagreement",
    "near_threshold",
    "large_score_gap",
    "selected_final",
    "rejection_reason",
    "error_message",
]

SHOT_SUMMARY_FIELDS = [
    "shot",
    "n_total_files",
    "n_failed_load",
    "n_nan_or_invalid",
    "n_tae_like",
    "n_mixed",
    "n_eae_like",
    "n_sent_to_classifiers",
    "n_gold_good",
    "n_silver_good",
    "n_final_good",
    "n_final_bad",
    "n_flagged",
    "fraction_good",
    "fraction_flagged",
    "rf_cnn_agreement_fraction",
    "mean_p_rf_good",
    "mean_p_cnn_good",
    "median_p_rf_good",
    "median_p_cnn_good",
    "mode_clusters_removed",
    "fraction_tae_threshold",
    "fraction_eae_threshold",
    "signed_delta_eae_threshold",
    "include_mixed_in_tae_like",
    "gold_good_threshold",
    "silver_good_threshold",
    "gold_bad_threshold",
    "silver_bad_threshold",
    "fallback_good_threshold",
]

SUMMARY_BY_N_FIELDS = ["shot", "n", *[f for f in SHOT_SUMMARY_FIELDS if f != "shot"]]

INVALID_REASON_NAMES = {
    "nonfinite_mode_data",
    "too_small_nhar",
    "ntor-N_mismatch",
    "invalid_metadata",
    "invalid_weights",
    "invalid_continuum",
    "invalid_upper2_scalars",
}


def parse_args() -> argparse.Namespace:
    ap = argparse.ArgumentParser(
        description=(
            "Process one mixed TAE/EAE NOVA shot, route EAE-like modes away, "
            "score TAE-like modes with RF + raw CNN, and write QC outputs."
        )
    )
    ap.add_argument("--shot_dir", required=True, help="Shot directory containing N1, N2, ...")
    ap.add_argument("--rf_model", required=True, help="RF classifier .joblib path")
    ap.add_argument("--cnn_model", required=True, help="Raw-CNN checkpoint .pt path")
    ap.add_argument("--out_dir", required=True, help="Directory for CSV outputs and reports")
    ap.add_argument("--device", default=None, help="Torch device for CNN inference, e.g. cpu or cuda")
    ap.add_argument("--n_min", type=int, default=1, help="Smallest N directory to scan")
    ap.add_argument("--n_max", type=int, default=10, help="Largest N directory to scan")
    ap.add_argument("--pattern", default="egn*", help="Mode-file glob within each N# directory")
    ap.add_argument("--rel_freq_tol", type=float, default=0.02, help="Relative frequency tolerance for duplicate clustering")
    ap.add_argument("--make_plots", action="store_true", help="Write optional QC plots")
    ap.add_argument("--verbose", action="store_true", help="Print per-directory progress")

    # Gap routing policy.
    ap.add_argument("--fraction_tae_threshold", type=float, default=0.5)
    ap.add_argument("--fraction_eae_threshold", type=float, default=0.4)
    ap.add_argument("--signed_delta_eae_threshold", type=float, default=-0.1)

    # RF/CNN fusion policy.
    ap.add_argument("--gold_good_threshold", type=float, default=0.8)
    ap.add_argument("--silver_good_threshold", type=float, default=0.6)
    ap.add_argument("--gold_bad_threshold", type=float, default=0.2)
    ap.add_argument("--silver_bad_threshold", type=float, default=0.4)
    ap.add_argument("--fallback_good_threshold", type=float, default=0.5)
    return ap.parse_args()


def make_base_row(path: str, shot: str, n: int) -> dict[str, Any]:
    row: dict[str, Any] = {field: "" for field in ALL_OUTPUT_FIELDS}
    row.update(
        {
            "path": path,
            "shot": shot,
            "n": int(n),
            "status": "",
            "selected_final": "",
        }
    )
    return row


def row_sort_key(row: dict[str, Any]) -> tuple[Any, ...]:
    n = row.get("n")
    omega = row.get("omega")
    ntor = row.get("ntor")
    return (
        int(n) if isinstance(n, (int, np.integer)) else math.inf,
        int(ntor) if isinstance(ntor, (int, np.integer)) else math.inf,
        float(omega) if isinstance(omega, (int, float, np.floating)) else math.inf,
        str(row.get("path", "")),
    )


def mark_rejected(row: dict[str, Any], reason: str, error_message: str) -> dict[str, Any]:
    row["status"] = "rejected"
    row["rejection_reason"] = reason
    row["error_message"] = error_message
    return row


def preflight_n_dirs(
    shot_dir: Path,
    *,
    n_min: int,
    n_max: int,
    pattern: str,
) -> list[tuple[int, Path, list[Path]]]:
    """
    Find populated N# directories and ensure the expected datcon# file exists
    and is readable before processing any modes from the shot.
    """
    populated: list[tuple[int, Path, list[Path]]] = []
    for n, ndir in iter_n_dirs(str(shot_dir), n_min=n_min, n_max=n_max):
        files = sorted(ndir.glob(pattern))
        if not files:
            continue

        datcon_path = ndir / f"datcon{n}"
        if not datcon_path.is_file():
            raise SystemExit(
                f"Cannot process shot: required continuum file is missing for N{n}: {datcon_path}"
            )
        try:
            with open(datcon_path, "r"):
                pass
        except OSError as exc:
            raise SystemExit(
                f"Cannot process shot: required continuum file is unreadable for N{n}: "
                f"{datcon_path} ({type(exc).__name__}: {exc})"
            ) from exc

        populated.append((n, ndir, files))

    return populated


def inspect_mode_file(path: str, *, expected_n: int) -> tuple[dict[str, Any] | None, str, str]:
    try:
        mode, omega, gamma_d, ntor = load_mode_from_nova(path)
    except Exception as exc:
        return None, "mode_load_failed", f"{type(exc).__name__}: {exc}"

    mode = np.asarray(mode)
    if mode.ndim != 2 or mode.shape[0] <= 0 or mode.shape[1] <= 0:
        return None, "invalid_metadata", f"Unexpected mode shape {mode.shape}"

    nhar, nr = mode.shape
    metadata_values = [omega, gamma_d, ntor, nr, nhar]
    if not all(np.isfinite(value) for value in metadata_values):
        return None, "invalid_metadata", "omega/gamma_d/ntor/nr/nhar contains a non-finite value"
    if int(ntor) != int(expected_n):
        return (
            None,
            "ntor-N_mismatch",
            f"Directory N{expected_n} disagrees with file metadata ntor={ntor}",
        )
    if nhar < 4 * int(ntor):
        return (
            None,
            "too_small_nhar",
            f"nhar={nhar} is smaller than required minimum 4*ntor={4 * int(ntor)}",
        )
    if not np.all(np.isfinite(mode)):
        return None, "nonfinite_mode_data", "Mode array contains NaN or non-finite values"

    weight_sum = float(np.sum(np.abs(mode) ** 2))
    if not np.isfinite(weight_sum) or weight_sum <= 0.0:
        return None, "invalid_weights", f"Invalid amplitude-squared weight sum: {weight_sum}"

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


def load_gap_scalars(
    path: str,
    *,
    mode: np.ndarray,
    omega: float,
) -> tuple[dict[str, float] | None, str, str]:
    try:
        _low2_full, upper2_full, *_ = load_datcon_for_mode(path, n_r=mode.shape[1])
    except Exception as exc:
        return None, "invalid_continuum", f"{type(exc).__name__}: {exc}"

    if not np.any(np.isfinite(upper2_full)):
        return None, "invalid_continuum", "upper2 continuum array has no finite values"

    try:
        scalars = upper2_scalars(mode, omega, upper2_full)
    except Exception as exc:
        return None, "invalid_upper2_scalars", f"{type(exc).__name__}: {exc}"

    return scalars, "", ""


def classify_gap_region(
    signed_delta: float,
    fraction_below_upper2: float,
    *,
    fraction_tae_threshold: float,
    fraction_eae_threshold: float,
    signed_delta_eae_threshold: float,
) -> str:
    if fraction_below_upper2 > fraction_tae_threshold:
        return "tae_like"
    if (
        fraction_below_upper2 < fraction_eae_threshold
        and signed_delta < signed_delta_eae_threshold
    ):
        return "eae_like"
    return "mixed"


def fuse_scores(
    p_rf_good: float,
    p_cnn_good: float,
    *,
    gold_good_threshold: float,
    silver_good_threshold: float,
    gold_bad_threshold: float,
    silver_bad_threshold: float,
    fallback_good_threshold: float,
) -> dict[str, Any]:
    p_avg = 0.5 * (float(p_rf_good) + float(p_cnn_good))

    if p_rf_good >= gold_good_threshold and p_cnn_good >= gold_good_threshold:
        final_label = "good"
        tier = "gold_good"
    elif p_rf_good <= gold_bad_threshold and p_cnn_good <= gold_bad_threshold:
        final_label = "bad"
        tier = "gold_bad"
    elif p_rf_good >= silver_good_threshold and p_cnn_good >= silver_good_threshold:
        final_label = "good"
        tier = "silver_good"
    elif p_rf_good <= silver_bad_threshold and p_cnn_good <= silver_bad_threshold:
        final_label = "bad"
        tier = "silver_bad"
    else:
        final_label = "good" if p_avg >= fallback_good_threshold else "bad"
        tier = "flagged_borderline_or_disagreement"

    return {
        "p_avg": p_avg,
        "final_label": final_label,
        "tier": tier,
        "model_disagreement": (p_rf_good - 0.5) * (p_cnn_good - 0.5) < 0.0,
        "near_threshold": abs(p_rf_good - 0.5) < 0.1 or abs(p_cnn_good - 0.5) < 0.1,
        "large_score_gap": abs(p_rf_good - p_cnn_good) > 0.4,
    }


def is_flagged_row(row: dict[str, Any]) -> bool:
    return bool(
        row.get("tier") == "flagged_borderline_or_disagreement"
        or row.get("model_disagreement") is True
        or row.get("near_threshold") is True
        or row.get("large_score_gap") is True
    )


def write_dict_rows_csv(
    path: Path,
    fieldnames: Sequence[str],
    rows: Sequence[dict[str, Any]],
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", newline="") as fp:
        writer = csv.DictWriter(fp, fieldnames=fieldnames, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)


def build_good_mode_dicts(
    rows: Sequence[dict[str, Any]],
    mode_cache: dict[str, np.ndarray],
    *,
    dm_band: int = 1,
    center_power: float = 2.0,
    median_k: int = 3,
    max_step: int = 2,
) -> list[dict[str, Any]]:
    good_modes: list[dict[str, Any]] = []
    for row in rows:
        if row.get("status") != "scored" or row.get("final_label") != "good":
            continue
        mode = mode_cache[row["path"]]
        good_modes.append(
            build_mode_dict(
                path=row["path"],
                shot=row["shot"],
                ntor=row["ntor"],
                omega=row["omega"],
                score=row["p_avg"],
                mode=mode,
                dm_band=dm_band,
                center_power=center_power,
                median_k=median_k,
                max_step=max_step,
            )
        )
    return good_modes


def set_selected_final(rows: Iterable[dict[str, Any]], selected_modes: Sequence[dict[str, Any]]) -> None:
    selected_paths = {mode["path"] for mode in selected_modes}
    for row in rows:
        if row.get("status") == "scored" and row.get("final_label") == "good":
            row["selected_final"] = row["path"] in selected_paths


def finite_values(rows: Sequence[dict[str, Any]], field: str) -> np.ndarray:
    values = [
        float(row[field])
        for row in rows
        if isinstance(row.get(field), (int, float, np.integer, np.floating))
        and np.isfinite(row[field])
    ]
    return np.asarray(values, dtype=float)


def safe_fraction(numer: int, denom: int) -> float | str:
    return float(numer / denom) if denom else ""


def safe_mean(values: np.ndarray) -> float | str:
    return float(np.mean(values)) if values.size else ""


def safe_median(values: np.ndarray) -> float | str:
    return float(np.median(values)) if values.size else ""


def build_summary_row(
    rows: Sequence[dict[str, Any]],
    *,
    shot: str,
    selected_modes: Sequence[dict[str, Any]],
    thresholds: dict[str, Any],
) -> dict[str, Any]:
    scored_rows = [row for row in rows if row.get("status") == "scored"]
    p_rf_values = finite_values(scored_rows, "p_rf_good")
    p_cnn_values = finite_values(scored_rows, "p_cnn_good")
    n_final_good = sum(row.get("final_label") == "good" for row in scored_rows)
    n_final_bad = sum(row.get("final_label") == "bad" for row in scored_rows)
    n_flagged = sum(is_flagged_row(row) for row in scored_rows)
    good_before_postprocess = n_final_good

    summary = {
        "shot": shot,
        "n_total_files": len(rows),
        "n_failed_load": sum(row.get("rejection_reason") == "mode_load_failed" for row in rows),
        "n_nan_or_invalid": sum(row.get("rejection_reason") in INVALID_REASON_NAMES for row in rows),
        "n_tae_like": sum(row.get("gap_region") == "tae_like" for row in rows),
        "n_mixed": sum(row.get("gap_region") == "mixed" for row in rows),
        "n_eae_like": sum(row.get("gap_region") == "eae_like" for row in rows),
        "n_sent_to_classifiers": len(scored_rows),
        "n_gold_good": sum(row.get("tier") == "gold_good" for row in scored_rows),
        "n_silver_good": sum(row.get("tier") == "silver_good" for row in scored_rows),
        "n_final_good": n_final_good,
        "n_final_bad": n_final_bad,
        "n_flagged": n_flagged,
        "fraction_good": safe_fraction(n_final_good, len(scored_rows)),
        "fraction_flagged": safe_fraction(n_flagged, len(scored_rows)),
        "rf_cnn_agreement_fraction": safe_fraction(
            sum(not bool(row.get("model_disagreement")) for row in scored_rows),
            len(scored_rows),
        ),
        "mean_p_rf_good": safe_mean(p_rf_values),
        "mean_p_cnn_good": safe_mean(p_cnn_values),
        "median_p_rf_good": safe_median(p_rf_values),
        "median_p_cnn_good": safe_median(p_cnn_values),
        "mode_clusters_removed": good_before_postprocess - len(selected_modes),
    }
    summary.update(thresholds)
    return summary


def build_summary_by_n(
    rows: Sequence[dict[str, Any]],
    *,
    shot: str,
    selected_modes: Sequence[dict[str, Any]],
    thresholds: dict[str, Any],
) -> list[dict[str, Any]]:
    selected_by_n: dict[int, list[dict[str, Any]]] = defaultdict(list)
    for mode in selected_modes:
        selected_by_n[int(mode["ntor"])].append(mode)

    rows_by_n: dict[int, list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        if isinstance(row.get("n"), (int, np.integer)):
            rows_by_n[int(row["n"])].append(row)

    summary_rows: list[dict[str, Any]] = []
    for n in sorted(rows_by_n):
        summary = build_summary_row(
            rows_by_n[n],
            shot=shot,
            selected_modes=selected_by_n.get(n, []),
            thresholds=thresholds,
        )
        summary_rows.append({"shot": shot, "n": n, **summary})
    return summary_rows


def write_outputs(
    rows: list[dict[str, Any]],
    *,
    out_dir: Path,
    selected_modes: Sequence[dict[str, Any]],
    cluster_records: Sequence[dict[str, Any]],
    summary_row: dict[str, Any],
    summary_by_n_rows: Sequence[dict[str, Any]],
    rel_freq_tol: float,
) -> None:
    rows_sorted = sorted(rows, key=row_sort_key)
    good_rows = [
        row for row in rows_sorted if row.get("status") == "scored" and row.get("final_label") == "good"
    ]
    bad_rows = [
        row for row in rows_sorted if row.get("status") == "scored" and row.get("final_label") == "bad"
    ]
    flagged_rows = [
        row for row in rows_sorted if row.get("status") == "scored" and is_flagged_row(row)
    ]
    eae_rows = [row for row in rows_sorted if row.get("status") == "eae_like"]
    rejected_rows = [row for row in rows_sorted if row.get("status") == "rejected"]
    tae_like_rows = [
        row for row in rows_sorted if row.get("gap_region") in {"tae_like", "mixed"}
    ]
    selected_paths = {mode["path"] for mode in selected_modes}
    final_good_rows = [row for row in good_rows if row["path"] in selected_paths]

    write_dict_rows_csv(out_dir / "all_modes_scored.csv", ALL_OUTPUT_FIELDS, rows_sorted)
    write_dict_rows_csv(out_dir / "tae_like_all.csv", ALL_OUTPUT_FIELDS, tae_like_rows)
    write_dict_rows_csv(out_dir / "good_tae_unchecked.csv", ALL_OUTPUT_FIELDS, good_rows)
    write_dict_rows_csv(out_dir / "good_tae_final.csv", ALL_OUTPUT_FIELDS, final_good_rows)
    write_dict_rows_csv(out_dir / "bad_tae_like.csv", ALL_OUTPUT_FIELDS, bad_rows)
    write_dict_rows_csv(out_dir / "flagged_tae_like.csv", ALL_OUTPUT_FIELDS, flagged_rows)
    write_dict_rows_csv(out_dir / "eae_like.csv", ALL_OUTPUT_FIELDS, eae_rows)
    write_dict_rows_csv(out_dir / "rejected_modes.csv", ALL_OUTPUT_FIELDS, rejected_rows)
    write_dict_rows_csv(out_dir / "shot_summary.csv", SHOT_SUMMARY_FIELDS, [summary_row])
    write_dict_rows_csv(out_dir / "shot_summary_by_n.csv", SUMMARY_BY_N_FIELDS, summary_by_n_rows)

    write_cluster_report(
        out_dir / "frequency_cluster_report.txt",
        cluster_records,
        rel_freq_tol=rel_freq_tol,
        sim_threshold=0.90,
        r_tol=0.10,
        width_tol=0.05,
    )
    write_cluster_csv(out_dir / "frequency_clusters.csv", cluster_records)


def make_plots(rows: Sequence[dict[str, Any]], out_dir: Path) -> None:
    try:
        import matplotlib.pyplot as plt
    except Exception as exc:
        print(f"[plots] Could not import matplotlib; skipping plots: {type(exc).__name__}: {exc}")
        return

    scored_rows = [row for row in rows if row.get("status") == "scored"]
    if not scored_rows:
        print("[plots] No scored TAE-like modes available; skipping plots.")
        return

    by_n: dict[int, list[dict[str, Any]]] = defaultdict(list)
    for row in scored_rows:
        by_n[int(row["n"])].append(row)

    try:
        fig, axes = plt.subplots(len(by_n), 1, figsize=(7, max(3, 2.2 * len(by_n))), squeeze=False)
        for ax, n in zip(axes[:, 0], sorted(by_n)):
            vals = finite_values(by_n[n], "p_rf_good")
            ax.hist(vals, bins=20, range=(0.0, 1.0))
            ax.set_title(f"N{n}")
            ax.set_xlim(0.0, 1.0)
            ax.set_xlabel("p_rf_good")
        fig.tight_layout()
        fig.savefig(out_dir / "hist_p_rf_good_by_n.png", dpi=150)
        plt.close(fig)
    except Exception as exc:
        print(f"[plots] Failed RF histogram plot: {type(exc).__name__}: {exc}")

    try:
        fig, axes = plt.subplots(len(by_n), 1, figsize=(7, max(3, 2.2 * len(by_n))), squeeze=False)
        for ax, n in zip(axes[:, 0], sorted(by_n)):
            vals = finite_values(by_n[n], "p_cnn_good")
            ax.hist(vals, bins=20, range=(0.0, 1.0))
            ax.set_title(f"N{n}")
            ax.set_xlim(0.0, 1.0)
            ax.set_xlabel("p_cnn_good")
        fig.tight_layout()
        fig.savefig(out_dir / "hist_p_cnn_good_by_n.png", dpi=150)
        plt.close(fig)
    except Exception as exc:
        print(f"[plots] Failed CNN histogram plot: {type(exc).__name__}: {exc}")

    try:
        colors = {
            "gold_good": "#1b9e77",
            "silver_good": "#66a61e",
            "gold_bad": "#d95f02",
            "silver_bad": "#e7298a",
            "flagged_borderline_or_disagreement": "#7570b3",
        }
        fig, ax = plt.subplots(figsize=(6, 6))
        for tier in colors:
            tier_rows = [row for row in scored_rows if row.get("tier") == tier]
            if not tier_rows:
                continue
            ax.scatter(
                finite_values(tier_rows, "p_rf_good"),
                finite_values(tier_rows, "p_cnn_good"),
                s=18,
                alpha=0.75,
                label=tier,
                color=colors[tier],
            )
        ax.set_xlabel("p_rf_good")
        ax.set_ylabel("p_cnn_good")
        ax.set_xlim(0.0, 1.0)
        ax.set_ylim(0.0, 1.0)
        ax.legend(fontsize=8)
        fig.tight_layout()
        fig.savefig(out_dir / "rf_vs_cnn_pgood.png", dpi=150)
        plt.close(fig)
    except Exception as exc:
        print(f"[plots] Failed RF-vs-CNN scatter plot: {type(exc).__name__}: {exc}")

    try:
        ns = sorted(by_n)
        total = [len(by_n[n]) for n in ns]
        gold = [sum(row.get("tier") == "gold_good" for row in by_n[n]) for n in ns]
        silver = [sum(row.get("tier") == "silver_good" for row in by_n[n]) for n in ns]
        flagged = [sum(is_flagged_row(row) for row in by_n[n]) for n in ns]
        bad = [sum(row.get("final_label") == "bad" for row in by_n[n]) for n in ns]
        x = np.arange(len(ns))
        width = 0.16
        fig, ax = plt.subplots(figsize=(max(7, 0.7 * len(ns)), 4))
        ax.bar(x - 2 * width, total, width, label="total TAE-like")
        ax.bar(x - width, gold, width, label="gold_good")
        ax.bar(x, silver, width, label="silver_good")
        ax.bar(x + width, flagged, width, label="flagged")
        ax.bar(x + 2 * width, bad, width, label="bad")
        ax.set_xticks(x)
        ax.set_xticklabels([f"N{n}" for n in ns])
        ax.legend(fontsize=8)
        fig.tight_layout()
        fig.savefig(out_dir / "counts_by_n.png", dpi=150)
        plt.close(fig)
    except Exception as exc:
        print(f"[plots] Failed counts-by-N plot: {type(exc).__name__}: {exc}")

    try:
        split_rows = [
            row
            for row in rows
            if isinstance(row.get("signed_delta"), (int, float, np.integer, np.floating))
            and isinstance(row.get("fraction_below_upper2"), (int, float, np.integer, np.floating))
        ]
        colors = {"tae_like": "#1b9e77", "mixed": "#7570b3", "eae_like": "#d95f02"}
        fig, ax = plt.subplots(figsize=(6, 5))
        for region, color in colors.items():
            region_rows = [row for row in split_rows if row.get("gap_region") == region]
            if not region_rows:
                continue
            ax.scatter(
                finite_values(region_rows, "signed_delta"),
                finite_values(region_rows, "fraction_below_upper2"),
                s=18,
                alpha=0.75,
                label=region,
                color=color,
            )
        ax.set_xlabel("signed_delta")
        ax.set_ylabel("fraction_below_upper2")
        ax.legend(fontsize=8)
        fig.tight_layout()
        fig.savefig(out_dir / "gap_split_diagnostic.png", dpi=150)
        plt.close(fig)
    except Exception as exc:
        print(f"[plots] Failed gap-split diagnostic plot: {type(exc).__name__}: {exc}")


def main() -> None:
    args = parse_args()
    shot_dir = Path(args.shot_dir).expanduser().resolve()
    out_dir = Path(args.out_dir).expanduser()
    if not shot_dir.is_dir():
        raise SystemExit(f"Shot directory not found: {shot_dir}")

    shot = shot_dir.name
    populated_n_dirs = preflight_n_dirs(
        shot_dir,
        n_min=args.n_min,
        n_max=args.n_max,
        pattern=args.pattern,
    )

    rf_clf = joblib.load(args.rf_model)
    cnn_clf = load_cnn_classifier(args.cnn_model, device=args.device, model_kind="cnn_raw")

    if args.verbose:
        print(f"Shot: {shot_dir}")
        print(f"RF model: {args.rf_model}")
        print(f"CNN model: {args.cnn_model}")
        print(f"Output directory: {out_dir}")
        print(f"Found populated N directories: {[n for n, _ndir, _files in populated_n_dirs]}")

    rows: list[dict[str, Any]] = []
    mode_cache: dict[str, np.ndarray] = {}

    for n, ndir, files in populated_n_dirs:
        if args.verbose:
            print(f"N{n}: found {len(files)} files in {ndir}")

        for path_obj in files:
            path = str(path_obj)
            row = make_base_row(path, shot, n)

            bundle, reason, error = inspect_mode_file(path, expected_n=n)
            if bundle is None:
                rows.append(mark_rejected(row, reason, error))
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

            scalars, reason, error = load_gap_scalars(
                path,
                mode=bundle["mode"],
                omega=bundle["omega"],
            )
            if scalars is None:
                rows.append(mark_rejected(row, reason, error))
                continue

            row.update(scalars)
            gap_region = classify_gap_region(
                scalars["signed_delta"],
                scalars["fraction_below_upper2"],
                fraction_tae_threshold=args.fraction_tae_threshold,
                fraction_eae_threshold=args.fraction_eae_threshold,
                signed_delta_eae_threshold=args.signed_delta_eae_threshold,
            )
            row["gap_region"] = gap_region

            if gap_region == "eae_like":
                row["status"] = "eae_like"
                rows.append(row)
                continue
            try:
                p_rf_good, _rf_mode, _omega_rf, _gamma_rf, _ntor_rf = classify_mode_rf(rf_clf, path)
            except Exception as exc:
                rows.append(
                    mark_rejected(
                        row,
                        "rf_inference_failed",
                        f"{type(exc).__name__}: {exc}",
                    )
                )
                continue

            try:
                cnn_result = cnn_clf.predict(path)
                p_cnn_good = float(cnn_result["p_good"])
            except Exception as exc:
                rows.append(
                    mark_rejected(
                        row,
                        "cnn_inference_failed",
                        f"{type(exc).__name__}: {exc}",
                    )
                )
                continue

            row.update(
                {
                    "status": "scored",
                    "p_rf_good": float(p_rf_good),
                    "p_cnn_good": p_cnn_good,
                }
            )
            row.update(
                fuse_scores(
                    float(p_rf_good),
                    p_cnn_good,
                    gold_good_threshold=args.gold_good_threshold,
                    silver_good_threshold=args.silver_good_threshold,
                    gold_bad_threshold=args.gold_bad_threshold,
                    silver_bad_threshold=args.silver_bad_threshold,
                    fallback_good_threshold=args.fallback_good_threshold,
                )
            )
            rows.append(row)

            if row["final_label"] == "good":
                mode_cache[path] = bundle["mode"]

    good_modes = build_good_mode_dicts(rows, mode_cache)
    selected_modes, cluster_records = postprocess_good_modes(
        good_modes,
        rel_freq_tol=args.rel_freq_tol,
        sim_threshold=0.90,
        r_tol=0.10,
        width_tol=0.05,
    )
    set_selected_final(rows, selected_modes)

    thresholds = {
        "fraction_tae_threshold": args.fraction_tae_threshold,
        "fraction_eae_threshold": args.fraction_eae_threshold,
        "signed_delta_eae_threshold": args.signed_delta_eae_threshold,
        "include_mixed_in_tae_like": True,
        "gold_good_threshold": args.gold_good_threshold,
        "silver_good_threshold": args.silver_good_threshold,
        "gold_bad_threshold": args.gold_bad_threshold,
        "silver_bad_threshold": args.silver_bad_threshold,
        "fallback_good_threshold": args.fallback_good_threshold,
    }
    summary_row = build_summary_row(
        rows,
        shot=shot,
        selected_modes=selected_modes,
        thresholds=thresholds,
    )
    summary_by_n_rows = build_summary_by_n(
        rows,
        shot=shot,
        selected_modes=selected_modes,
        thresholds=thresholds,
    )
    write_outputs(
        rows,
        out_dir=out_dir,
        selected_modes=selected_modes,
        cluster_records=cluster_records,
        summary_row=summary_row,
        summary_by_n_rows=summary_by_n_rows,
        rel_freq_tol=args.rel_freq_tol,
    )
    if args.make_plots:
        make_plots(rows, out_dir)

    print("=== Mixed-shot summary ===")
    print(f"Shot: {shot}")
    print(f"Total files: {summary_row['n_total_files']}")
    print(
        "Gap split: "
        f"tae_like={summary_row['n_tae_like']} "
        f"mixed={summary_row['n_mixed']} "
        f"eae_like={summary_row['n_eae_like']}"
    )
    print(
        "Scored TAE-side modes: "
        f"{summary_row['n_sent_to_classifiers']} | "
        f"final_good={summary_row['n_final_good']} "
        f"final_bad={summary_row['n_final_bad']} "
        f"flagged={summary_row['n_flagged']}"
    )
    print(f"Final selected GOOD modes after clustering: {len(selected_modes)}")
    print(f"Wrote outputs to: {out_dir}")


if __name__ == "__main__":
    main()
