#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
from pathlib import Path

from cnn_infer_common import (
    SUPPORTED_MODEL_KINDS,
    UnsupportedCheckpointError,
    load_cnn_classifier,
    read_mode_paths_csv,
)
from path_utils import resolve_mode_csv_path


def parse_args() -> argparse.Namespace:
    ap = argparse.ArgumentParser(
        description="Classify NOVA modes with a straightened or hybrid CNN checkpoint."
    )
    ap.add_argument("path_pos", nargs="?", help="Mode file path (positional form: mode_file checkpoint.pt)")
    ap.add_argument("model_pos", nargs="?", help="Checkpoint path (positional form: mode_file checkpoint.pt)")
    ap.add_argument("--path", dest="path_opt", help="Mode file path to classify")
    ap.add_argument("--model", dest="model_opt", help="Checkpoint path (.pt)")
    ap.add_argument("--csv", dest="csv_in", help="CSV list of mode paths (first column is used)")
    ap.add_argument("--out", help="Output CSV path for --csv mode")
    ap.add_argument("--device", help="Torch device, e.g. cpu, cuda, cuda:0")
    ap.add_argument("--threshold", type=float, default=None, help="Override checkpoint threshold")
    ap.add_argument(
        "--model_kind",
        choices=["auto", *sorted(SUPPORTED_MODEL_KINDS)],
        default="auto",
        help="Force checkpoint type instead of auto-detecting it",
    )
    return ap.parse_args()


def main() -> None:
    args = parse_args()

    model_path = args.model_opt or args.model_pos
    mode_path = args.path_opt or args.path_pos

    if not model_path:
        raise SystemExit("Need a checkpoint path via positional arg or --model.")
    if bool(mode_path) == bool(args.csv_in):
        raise SystemExit("Provide exactly one of a single mode path or --csv.")

    try:
        classifier = load_cnn_classifier(
            model_path,
            device=args.device,
            model_kind=args.model_kind,
        )
    except UnsupportedCheckpointError as exc:
        raise SystemExit(str(exc))

    if args.csv_in:
        mode_paths = read_mode_paths_csv(args.csv_in)
        out_path = Path(args.out) if args.out else Path(args.csv_in).with_name(
            f"{Path(args.csv_in).stem}_preds.csv"
        )

        rows: list[list[str]] = []
        for path in mode_paths:
            try:
                result = classifier.predict(path, threshold=args.threshold)
                rows.append(
                    [
                        result["path"],
                        result["label"],
                        f"{result['p_good']:.6f}",
                        f"{result['omega']:.8g}",
                        f"{result['gamma_d']:.8g}",
                        str(result["ntor"]),
                        "",
                    ]
                )
            except Exception as exc:
                rows.append([path, "error", "", "", "", "", str(exc)])

        out_path.parent.mkdir(parents=True, exist_ok=True)
        with open(out_path, "w", newline="") as fp:
            writer = csv.writer(fp)
            writer.writerow(["path", "label", "p_good", "omega", "gamma_d", "ntor", "error"])
            writer.writerows(rows)

        print(f"Model: {classifier.checkpoint_path}")
        print(f"Detected checkpoint kind: {classifier.checkpoint_kind}")
        print(f"Wrote predictions to {out_path}")
        return

    result = classifier.predict(resolve_mode_csv_path(mode_path), threshold=args.threshold)

    print(f"{result['path']}")
    print(f"  checkpoint_kind={result['checkpoint_kind']}")
    print(f"  p_good={result['p_good']:.6f}  label={result['label']}  threshold={result['threshold']:.6f}")
    print(
        f"  omega={result['omega']:.8g}  gamma_d={result['gamma_d']:.8g}  ntor={result['ntor']}"
    )


if __name__ == "__main__":
    main()
