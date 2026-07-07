#!/usr/bin/env python3
"""
Run leave-one-shot-out checks for the expanded TAE-like training set.

The driver keeps small, versionable evaluation CSVs under outputs/loso_<N> by
default, while model checkpoints and training logs go to $NOVA_RUN/loso_<N> or
$SCRATCH/nova_s/loso_<N>. It intentionally calls the existing RF, raw-CNN, and
mixed-shot sorter scripts rather than duplicating their preprocessing logic.
"""

from __future__ import annotations

import argparse
import csv
import json
import os
import re
import subprocess
import sys
from collections import Counter, defaultdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable, Sequence


STEP_NAMES = ("split", "rf", "cnn", "sort", "aggregate")
LABEL_FIELDS = ("validity", "label", "manual_label", "target", "class")


def default_repo_root() -> Path:
    return Path(__file__).resolve().parents[1]


def default_out_root(repo_root: Path, n_shots: int) -> Path:
    return repo_root / "outputs" / f"loso_{n_shots}"


def default_work_root(out_root: Path) -> Path:
    run_name = out_root.name
    nova_run = os.environ.get("NOVA_RUN")
    if nova_run:
        return Path(nova_run).expanduser() / run_name

    scratch = os.environ.get("SCRATCH")
    if scratch:
        return Path(scratch).expanduser() / "nova_s" / run_name

    return out_root / "work"


def parse_steps(raw: str) -> list[str]:
    steps = [step.strip().lower() for step in raw.split(",") if step.strip()]
    if not steps or steps == ["all"]:
        return list(STEP_NAMES)

    unknown = [step for step in steps if step not in STEP_NAMES]
    if unknown:
        raise argparse.ArgumentTypeError(
            f"Unknown step(s): {', '.join(unknown)}. Valid steps: all, {', '.join(STEP_NAMES)}"
        )
    return steps


def parse_folds(raw: str | None) -> set[str] | None:
    if raw is None or not raw.strip():
        return None
    return {part.strip() for part in raw.split(",") if part.strip()}


def shot_from_mode_path(path_value: str) -> str:
    parts = Path(path_value.strip()).parts
    for idx, part in enumerate(parts[:-1]):
        if re.fullmatch(r"N\d+", parts[idx + 1]):
            return part

    if parts and parts[0] not in {"", os.sep}:
        return parts[0]

    raise ValueError(f"Could not infer shot from mode path: {path_value!r}")


def read_training_rows(csv_path: Path) -> tuple[list[str], list[dict[str, str]], str]:
    with csv_path.open("r", newline="") as fp:
        reader = csv.DictReader(fp)
        if not reader.fieldnames:
            raise ValueError(f"Training CSV must have a header row: {csv_path}")
        fieldnames = list(reader.fieldnames)
        rows = [dict(row) for row in reader]

    if "path" not in fieldnames:
        raise ValueError(f"Training CSV must contain a 'path' column: {csv_path}")

    label_field = next((field for field in LABEL_FIELDS if field in fieldnames), None)
    if label_field is None:
        raise ValueError(
            f"Training CSV must contain one label column from {LABEL_FIELDS}: {csv_path}"
        )

    return fieldnames, rows, label_field


def normalized_label(row: dict[str, str], label_field: str) -> str:
    return (row.get(label_field) or "").strip().lower()


def write_rows_csv(path: Path, fieldnames: Sequence[str], rows: Sequence[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="") as fp:
        writer = csv.DictWriter(fp, fieldnames=fieldnames, lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)


def count_labels(rows: Iterable[dict[str, str]], label_field: str) -> Counter[str]:
    counts: Counter[str] = Counter()
    for row in rows:
        label = normalized_label(row, label_field)
        if label in {"good", "bad"}:
            counts[label] += 1
        else:
            counts["other"] += 1
    return counts


def prepare_loso_splits(
    *,
    train_csv: Path,
    out_root: Path,
    selected_folds: set[str] | None,
) -> list[str]:
    fieldnames, rows, label_field = read_training_rows(train_csv)

    rows_by_shot: dict[str, list[dict[str, str]]] = defaultdict(list)
    for row in rows:
        rows_by_shot[shot_from_mode_path(row["path"])].append(row)

    all_shots = sorted(rows_by_shot)
    if selected_folds:
        missing = sorted(selected_folds - set(all_shots))
        if missing:
            raise ValueError(f"Requested fold shot(s) not present in {train_csv}: {missing}")
        fold_shots = [shot for shot in all_shots if shot in selected_folds]
    else:
        fold_shots = all_shots

    count_fields = ["fold", "shot", "split", "n_rows", "n_good", "n_bad", "n_other"]
    count_rows: list[dict[str, Any]] = []

    for fold_idx, shot in enumerate(fold_shots, 1):
        train_rows = [row for row in rows if shot_from_mode_path(row["path"]) != shot]
        test_rows = rows_by_shot[shot]
        fold_dir = out_root / "folds" / shot
        write_rows_csv(fold_dir / "train.csv", fieldnames, train_rows)
        write_rows_csv(fold_dir / "test.csv", fieldnames, test_rows)

        for split_name, split_rows in (("train", train_rows), ("test", test_rows)):
            labels = count_labels(split_rows, label_field)
            count_rows.append(
                {
                    "fold": fold_idx,
                    "shot": shot,
                    "split": split_name,
                    "n_rows": len(split_rows),
                    "n_good": labels.get("good", 0),
                    "n_bad": labels.get("bad", 0),
                    "n_other": labels.get("other", 0),
                }
            )

    write_rows_csv(out_root / "loso_split_counts.csv", count_fields, count_rows)
    manifest = {
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "train_csv": str(train_csv),
        "label_field": label_field,
        "n_rows": len(rows),
        "shots": all_shots,
        "fold_shots": fold_shots,
    }
    (out_root / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
    return fold_shots


def count_training_shots(train_csv: Path) -> int:
    _, rows, _ = read_training_rows(train_csv)
    return len({shot_from_mode_path(row["path"]) for row in rows})


def read_existing_fold_shots(out_root: Path) -> list[str]:
    folds_dir = out_root / "folds"
    if not folds_dir.is_dir():
        raise FileNotFoundError(
            f"No LOSO split directory found: {folds_dir}. Run with --steps split first."
        )
    return sorted(path.name for path in folds_dir.iterdir() if (path / "train.csv").is_file())


def split_counts(out_root: Path) -> dict[tuple[str, str], dict[str, str]]:
    path = out_root / "loso_split_counts.csv"
    if not path.is_file():
        return {}
    with path.open("r", newline="") as fp:
        return {(row["shot"], row["split"]): row for row in csv.DictReader(fp)}


def prepend_pythonpath(env: dict[str, str], repo_root: Path) -> None:
    src_path = str(repo_root / "src")
    current = env.get("PYTHONPATH", "")
    env["PYTHONPATH"] = src_path if not current else f"{src_path}{os.pathsep}{current}"


def srun_prefix(args: argparse.Namespace) -> list[str]:
    return [
        "srun",
        "--nodes",
        "1",
        "--ntasks",
        "1",
        "--cpus-per-task",
        str(args.cpus_per_task),
        "--gpus-per-task",
        "1",
        "--gpu-bind=none",
        "--kill-on-bad-exit=1",
    ]


def resolve_launch_prefix(args: argparse.Namespace, launch: str) -> list[str]:
    if launch == "plain":
        return []
    if launch == "srun":
        return srun_prefix(args)
    if launch == "auto" and os.environ.get("SLURM_JOB_ID"):
        return srun_prefix(args)
    return []


def command_to_string(cmd: Sequence[str]) -> str:
    return " ".join(subprocess.list2cmdline([part]) for part in cmd)


def write_run_config(args: argparse.Namespace, fold_shots: Sequence[str]) -> None:
    config = {
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "repo_root": str(args.repo_root),
        "train_csv": str(args.train_csv),
        "out_root": str(args.out_root),
        "work_root": str(args.work_root),
        "data_dir": str(args.data_dir),
        "shot_root": str(args.shot_root),
        "steps": args.steps,
        "folds": list(fold_shots),
        "n_folds": len(fold_shots),
        "seed": args.seed,
        "cnn_m_target": args.cnn_m_target,
        "cnn_r_target": args.cnn_r_target,
        "cnn_epochs": args.cnn_epochs,
        "cnn_batch_size": args.cnn_batch_size,
        "cnn_lr": args.cnn_lr,
        "cnn_test_frac": args.cnn_test_frac,
        "cnn_normalize": args.cnn_normalize,
        "cnn_pos_weight": args.cnn_pos_weight,
        "cnn_refit_full_before_save": args.cnn_refit_full_before_save,
        "model_eval_threshold": args.model_eval_threshold,
    }
    (args.out_root / "run_config.json").write_text(json.dumps(config, indent=2) + "\n")


def run_logged(
    cmd: Sequence[str],
    *,
    log_path: Path,
    cwd: Path,
    env: dict[str, str],
    dry_run: bool,
) -> None:
    log_path.parent.mkdir(parents=True, exist_ok=True)
    header = [
        f"started_utc={datetime.now(timezone.utc).isoformat()}",
        f"cwd={cwd}",
        f"command={command_to_string(cmd)}",
        "",
    ]
    with log_path.open("w") as log_fp:
        log_fp.write("\n".join(header))
        log_fp.flush()
        if dry_run:
            log_fp.write("[dry-run] command not executed\n")
            return

        result = subprocess.run(
            list(cmd),
            cwd=str(cwd),
            env=env,
            stdout=log_fp,
            stderr=subprocess.STDOUT,
            text=True,
            check=False,
        )
        log_fp.write(f"\nfinished_utc={datetime.now(timezone.utc).isoformat()}\n")
        log_fp.write(f"returncode={result.returncode}\n")
        if result.returncode != 0:
            raise subprocess.CalledProcessError(result.returncode, list(cmd))


def fold_paths(args: argparse.Namespace, shot: str) -> dict[str, Path]:
    out_fold = args.out_root / "folds" / shot
    work_fold = args.work_root / "folds" / shot
    return {
        "train_csv": out_fold / "train.csv",
        "test_csv": out_fold / "test.csv",
        "rf_model": work_fold / "models" / "nova_mode_classifier.joblib",
        "cnn_model": work_fold / "models" / "nova_cnn_raw.pt",
        "sort_dir": out_fold / "sort_shot_mixed",
        "rf_log": work_fold / "logs" / "rf_train.log",
        "cnn_log": work_fold / "logs" / "cnn_raw_train.log",
        "sort_log": work_fold / "logs" / "sort_shot_mixed.log",
    }


def run_rf_fold(args: argparse.Namespace, shot: str, env: dict[str, str]) -> None:
    paths = fold_paths(args, shot)
    if args.skip_existing and paths["rf_model"].is_file():
        print(f"[rf] {shot}: found {paths['rf_model']}, skipping")
        return
    paths["rf_model"].parent.mkdir(parents=True, exist_ok=True)
    paths["rf_log"].parent.mkdir(parents=True, exist_ok=True)
    cmd = [
        args.python,
        str(args.repo_root / "scripts" / "rf_train_classify.py"),
        "--train_csv",
        str(paths["train_csv"]),
        "--model_out",
        str(paths["rf_model"]),
    ]
    print(f"[rf] {shot}: training -> {paths['rf_model']}")
    run_logged(cmd, log_path=paths["rf_log"], cwd=paths["rf_model"].parent, env=env, dry_run=args.dry_run)


def run_cnn_fold(args: argparse.Namespace, shot: str, env: dict[str, str]) -> None:
    paths = fold_paths(args, shot)
    if args.skip_existing and paths["cnn_model"].is_file():
        print(f"[cnn] {shot}: found {paths['cnn_model']}, skipping")
        return
    paths["cnn_model"].parent.mkdir(parents=True, exist_ok=True)
    cmd = [
        *resolve_launch_prefix(args, args.cnn_launch),
        args.python,
        "-u",
        str(args.repo_root / "scripts" / "cnn_raw.py"),
        "--train_csv",
        str(paths["train_csv"]),
        "--data_dir",
        str(args.data_dir),
        "--model_out",
        str(paths["cnn_model"]),
        "--test_frac",
        str(args.cnn_test_frac),
        "--seed",
        str(args.seed),
        "--batch_size",
        str(args.cnn_batch_size),
        "--epochs",
        str(args.cnn_epochs),
        "--lr",
        str(args.cnn_lr),
        "--normalize",
        args.cnn_normalize,
        "--eval_threshold",
        str(args.model_eval_threshold),
        "--M_target",
        str(args.cnn_m_target),
        "--R_target",
        str(args.cnn_r_target),
        "--device",
        args.cnn_device,
    ]
    if args.cnn_pos_weight:
        cmd.extend(["--pos_weight", args.cnn_pos_weight])
    if args.cnn_cache_data:
        cmd.append("--cache_data")
    if args.cnn_refit_full_before_save:
        cmd.append("--refit_full_before_save")

    print(f"[cnn] {shot}: training -> {paths['cnn_model']}")
    run_logged(cmd, log_path=paths["cnn_log"], cwd=args.repo_root, env=env, dry_run=args.dry_run)


def run_sort_fold(args: argparse.Namespace, shot: str, env: dict[str, str]) -> None:
    paths = fold_paths(args, shot)
    summary_path = paths["sort_dir"] / "model_evaluation_summary.csv"
    if args.skip_existing and summary_path.is_file():
        print(f"[sort] {shot}: found {summary_path}, skipping")
        return

    shot_dir = args.shot_root / shot
    cmd = [
        *resolve_launch_prefix(args, args.sort_launch),
        args.python,
        "-u",
        str(args.repo_root / "scripts" / "sort_shot_mixed.py"),
        "--shot_dir",
        str(shot_dir),
        "--rf_model",
        str(paths["rf_model"]),
        "--cnn_model",
        str(paths["cnn_model"]),
        "--cnn_model_kind",
        "cnn_raw",
        "--out_dir",
        str(paths["sort_dir"]),
        "--label_csv",
        str(paths["test_csv"]),
        "--model_eval_threshold",
        str(args.model_eval_threshold),
        "--device",
        args.sort_device,
        "--n_min",
        str(args.n_min),
        "--n_max",
        str(args.n_max),
        "--pattern",
        args.pattern,
    ]
    if args.make_plots:
        cmd.append("--make_plots")

    print(f"[sort] {shot}: evaluating -> {paths['sort_dir']}")
    run_logged(cmd, log_path=paths["sort_log"], cwd=args.repo_root, env=env, dry_run=args.dry_run)


def read_csv_rows(path: Path) -> list[dict[str, str]]:
    if not path.is_file():
        return []
    with path.open("r", newline="") as fp:
        return [dict(row) for row in csv.DictReader(fp)]


def to_int(value: str | int | float | None) -> int:
    if value in (None, ""):
        return 0
    return int(float(value))


def safe_fraction(numer: int, denom: int) -> float | str:
    return float(numer / denom) if denom else ""


def good_metrics(tn: int, fp: int, fn: int, tp: int) -> dict[str, float | str]:
    accuracy = safe_fraction(tn + tp, tn + fp + fn + tp)
    precision = safe_fraction(tp, tp + fp)
    recall = safe_fraction(tp, tp + fn)
    if isinstance(precision, float) and isinstance(recall, float) and precision + recall > 0.0:
        f1: float | str = 2.0 * precision * recall / (precision + recall)
    else:
        f1 = ""
    return {
        "accuracy": accuracy,
        "precision_good": precision,
        "recall_good": recall,
        "f1_good": f1,
    }


def aggregate_outputs(args: argparse.Namespace, fold_shots: Sequence[str]) -> None:
    counts = split_counts(args.out_root)
    eval_fields = [
        "shot",
        "n_train",
        "n_train_good",
        "n_train_bad",
        "n_test",
        "n_test_good",
        "n_test_bad",
        "model",
        "n_matched",
        "tn_bad",
        "fp_bad_as_good",
        "fn_good_as_bad",
        "tp_good",
        "accuracy",
        "precision_good",
        "recall_good",
        "f1_good",
    ]
    eval_rows: list[dict[str, Any]] = []
    missing_rows: list[dict[str, str]] = []
    totals: dict[str, Counter[str]] = defaultdict(Counter)

    shot_summary_rows: list[dict[str, Any]] = []
    shot_summary_fields: list[str] = []

    for shot in fold_shots:
        paths = fold_paths(args, shot)
        train_counts = counts.get((shot, "train"), {})
        test_counts = counts.get((shot, "test"), {})

        metric_path = paths["sort_dir"] / "model_evaluation_summary.csv"
        metric_rows = read_csv_rows(metric_path)
        if not metric_rows:
            missing_rows.append({"shot": shot, "missing": str(metric_path)})
        for row in metric_rows:
            out_row: dict[str, Any] = {
                "shot": shot,
                "n_train": train_counts.get("n_rows", ""),
                "n_train_good": train_counts.get("n_good", ""),
                "n_train_bad": train_counts.get("n_bad", ""),
                "n_test": test_counts.get("n_rows", ""),
                "n_test_good": test_counts.get("n_good", ""),
                "n_test_bad": test_counts.get("n_bad", ""),
            }
            out_row.update(row)
            eval_rows.append(out_row)

            model = row["model"]
            for field in ("n_matched", "tn_bad", "fp_bad_as_good", "fn_good_as_bad", "tp_good"):
                totals[model][field] += to_int(row.get(field))

        summary_rows = read_csv_rows(paths["sort_dir"] / "shot_summary_wide.csv")
        if summary_rows:
            summary = {"shot": shot, **summary_rows[0]}
            shot_summary_rows.append(summary)
            for field in summary:
                if field not in shot_summary_fields:
                    shot_summary_fields.append(field)

    write_rows_csv(args.out_root / "loso_model_evaluation_summary.csv", eval_fields, eval_rows)

    total_fields = [
        "model",
        "n_folds",
        "n_matched",
        "tn_bad",
        "fp_bad_as_good",
        "fn_good_as_bad",
        "tp_good",
        "accuracy",
        "precision_good",
        "recall_good",
        "f1_good",
    ]
    total_rows: list[dict[str, Any]] = []
    for model in sorted(totals):
        counts_for_model = totals[model]
        metrics = good_metrics(
            counts_for_model["tn_bad"],
            counts_for_model["fp_bad_as_good"],
            counts_for_model["fn_good_as_bad"],
            counts_for_model["tp_good"],
        )
        total_rows.append(
            {
                "model": model,
                "n_folds": sum(1 for row in eval_rows if row.get("model") == model),
                "n_matched": counts_for_model["n_matched"],
                "tn_bad": counts_for_model["tn_bad"],
                "fp_bad_as_good": counts_for_model["fp_bad_as_good"],
                "fn_good_as_bad": counts_for_model["fn_good_as_bad"],
                "tp_good": counts_for_model["tp_good"],
                **metrics,
            }
        )
    write_rows_csv(args.out_root / "loso_model_evaluation_totals.csv", total_fields, total_rows)

    if shot_summary_rows:
        write_rows_csv(args.out_root / "loso_shot_summary.csv", shot_summary_fields, shot_summary_rows)
    if missing_rows:
        write_rows_csv(args.out_root / "loso_missing_outputs.csv", ["shot", "missing"], missing_rows)

    print(f"[aggregate] wrote {args.out_root / 'loso_model_evaluation_summary.csv'}")
    print(f"[aggregate] wrote {args.out_root / 'loso_model_evaluation_totals.csv'}")


def build_arg_parser() -> argparse.ArgumentParser:
    repo_root = default_repo_root()

    ap = argparse.ArgumentParser(
        description=(
            "Create LOSO splits for all shots in the training CSV, retrain RF "
            "and raw CNN per fold, "
            "run sort_shot_mixed.py on held-out shots, and aggregate metrics."
        )
    )
    ap.add_argument("--repo_root", type=Path, default=repo_root)
    ap.add_argument("--train_csv", type=Path, default=repo_root / "training_labels" / "tae_like_train.csv")
    ap.add_argument(
        "--out_root",
        type=Path,
        default=None,
        help="Output root for split/evaluation CSVs (default: outputs/loso_<N shots>)",
    )
    ap.add_argument(
        "--work_root",
        type=Path,
        default=None,
        help=(
            "Root for checkpoints/logs (default: $NOVA_RUN/<out name>, "
            "$SCRATCH/nova_s/<out name>, or <out_root>/work)"
        ),
    )
    ap.add_argument("--data_dir", type=Path, default=Path(os.environ.get("NOVA_DATA", "")) if os.environ.get("NOVA_DATA") else None)
    ap.add_argument("--shot_root", type=Path, default=None, help="Root containing held-out shot directories (default: --data_dir)")
    ap.add_argument("--steps", type=parse_steps, default=list(STEP_NAMES), help="Comma list: all, split, rf, cnn, sort, aggregate")
    ap.add_argument("--folds", default=None, help="Optional comma-separated shot names to run")
    ap.add_argument("--python", default=sys.executable, help="Python executable for subprocesses")
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--skip_existing", action="store_true", help="Skip RF/CNN/sort steps when their expected output already exists")
    ap.add_argument("--dry_run", action="store_true", help="Write split files and logs but do not execute RF/CNN/sort commands")

    ap.add_argument("--cnn_launch", choices=["auto", "plain", "srun"], default="auto", help="How to launch CNN training commands")
    ap.add_argument("--sort_launch", choices=["auto", "plain", "srun"], default="plain", help="How to launch sort_shot_mixed.py")
    ap.add_argument("--cpus_per_task", type=int, default=int(os.environ.get("NOVA_CPUS_PER_TASK", "1")))
    ap.add_argument("--cnn_device", default="cuda", help="Torch device for CNN training; default requires a GPU")
    ap.add_argument("--sort_device", default="cpu", help="Torch device for CNN inference inside sort_shot_mixed.py")
    ap.add_argument("--cnn_epochs", type=int, default=80)
    ap.add_argument("--cnn_batch_size", type=int, default=8)
    ap.add_argument("--cnn_lr", type=float, default=2e-2)
    ap.add_argument("--cnn_test_frac", type=float, default=0.2)
    ap.add_argument("--cnn_normalize", choices=["none", "standard", "robust", "maxabs"], default="robust")
    ap.add_argument("--cnn_pos_weight", default=None, help="Pass through to cnn_raw.py, e.g. auto")
    ap.add_argument("--cnn_m_target", type=int, default=54)
    ap.add_argument("--cnn_r_target", type=int, default=201)
    ap.add_argument("--cnn_cache_data", action="store_true")
    ap.add_argument(
        "--cnn_refit_full_before_save",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Refit raw CNN on the full LOSO train fold after epoch selection",
    )

    ap.add_argument("--model_eval_threshold", type=float, default=0.5)
    ap.add_argument("--n_min", type=int, default=1)
    ap.add_argument("--n_max", type=int, default=10)
    ap.add_argument("--pattern", default="egn*")
    ap.add_argument("--make_plots", action="store_true")
    return ap


def main() -> None:
    args = build_arg_parser().parse_args()
    args.repo_root = args.repo_root.expanduser().resolve()
    args.train_csv = args.train_csv.expanduser().resolve()
    n_train_shots = count_training_shots(args.train_csv)
    if args.out_root is None:
        args.out_root = default_out_root(args.repo_root, n_train_shots)
    args.out_root = args.out_root.expanduser().resolve()
    if args.work_root is None:
        args.work_root = default_work_root(args.out_root)
    args.work_root = args.work_root.expanduser().resolve()
    if args.data_dir is None:
        raise SystemExit("--data_dir is required when $NOVA_DATA is not set")
    args.data_dir = args.data_dir.expanduser().resolve()
    args.shot_root = (args.shot_root or args.data_dir).expanduser().resolve()

    selected_folds = parse_folds(args.folds)
    args.out_root.mkdir(parents=True, exist_ok=True)
    args.work_root.mkdir(parents=True, exist_ok=True)

    env = os.environ.copy()
    prepend_pythonpath(env, args.repo_root)
    env["NOVA_DATA"] = str(args.data_dir)
    env.setdefault("NOVA_TORCH_DEVICE", args.cnn_device)

    print(f"repo_root: {args.repo_root}")
    print(f"train_csv: {args.train_csv}")
    print(f"out_root:  {args.out_root}")
    print(f"work_root: {args.work_root}")
    print(f"data_dir:  {args.data_dir}")
    print(f"steps:     {', '.join(args.steps)}")
    print(f"n_shots:   {n_train_shots}")
    print(f"M_target:  {args.cnn_m_target}")

    fold_shots: list[str]
    if "split" in args.steps:
        fold_shots = prepare_loso_splits(
            train_csv=args.train_csv,
            out_root=args.out_root,
            selected_folds=selected_folds,
        )
    else:
        fold_shots = read_existing_fold_shots(args.out_root)
        if selected_folds:
            fold_shots = [shot for shot in fold_shots if shot in selected_folds]

    print(f"folds:     {', '.join(fold_shots)}")
    write_run_config(args, fold_shots)

    for shot in fold_shots:
        if "rf" in args.steps:
            run_rf_fold(args, shot, env)
        if "cnn" in args.steps:
            run_cnn_fold(args, shot, env)
        if "sort" in args.steps:
            run_sort_fold(args, shot, env)

    if "aggregate" in args.steps:
        aggregate_outputs(args, fold_shots)


if __name__ == "__main__":
    main()
