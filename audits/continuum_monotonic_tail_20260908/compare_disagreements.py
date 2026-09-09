"""Build the incremental review list after the shared continuum repair.

Run from any directory with Python (standard library only):
  python audits/continuum_monotonic_tail_20260908/compare_disagreements.py \
    --out-dir audits/continuum_monotonic_tail_20260908/disagreement_delta_20260909

Old/new export locations come from publication.json. The edited question list
is read only; exclusion as already approved requires a matching fingerprint
and an explicit review decision matching the current rules decision.
"""

import argparse
from collections import Counter
import csv
import hashlib
import json
from pathlib import Path

HERE = Path(__file__).resolve().parent
REPO = HERE.parents[1]
DECISION_FIELDS = (
    "rules_decision",
    "rules_reason",
    "rf_cnn_decision",
    "input_fingerprint",
)


def read(path):
    with path.open() as handle:
        return list(csv.DictReader(handle))


def keyed(rows):
    result = {row["mode_key"]: row for row in rows}
    assert len(result) == len(rows), "Duplicate mode keys"
    return result


def write(path, fields, rows):
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields, lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out-dir", type=Path, required=True)
    args = parser.parse_args()
    inputs = {
        "current": HERE / "regenerated_disagreements.csv",
        "previous_12": REPO / "audits/pilot12_v5_20260908/disagreements.csv",
        "previous_15": REPO / "audits/regression15_v5/rf_cnn_disagreements.csv",
        "approved": HERE / "user_review.csv",
        "questions": REPO / "audits/pilot12_v5_20260908/disagreements_elena.csv",
        "exports": HERE / "publication.json",
    }
    hashes = {
        str(p.relative_to(REPO)): hashlib.sha256(p.read_bytes()).hexdigest()
        for p in inputs.values()
    }
    current = keyed(read(inputs["current"]))
    previous = keyed(read(inputs["previous_12"]))
    prior15 = read(inputs["previous_15"])
    for row in prior15:
        assert row["mode_key"] not in previous
        previous[row["mode_key"]] = dict(
            row, rules_decision=row["v5_decision"], rules_reason=row["v5_reason"]
        )
    assert all(r["rules_decision"] != r["rf_cnn_decision"] for r in previous.values())
    reviews = keyed(read(inputs["approved"]))
    questions = keyed(read(inputs["questions"]))
    exports = {
        (r["shot"], r["method"]): r for r in json.loads(inputs["exports"].read_text())
    }
    cache = {}

    def state(key, period):
        shot = key.split("/")[0]
        values = {}
        for method, filename in [
            ("rules", "all_modes_rules.csv"),
            ("rf-cnn", "all_modes_scored.csv"),
        ]:
            path = Path(exports[(shot, method)][period]) / filename
            if path not in cache:
                rows = read(path)
                cache[path] = {"/".join(Path(r["path"]).parts[-3:]): r for r in rows}
                hashes[str(path)] = hashlib.sha256(path.read_bytes()).hexdigest()
            row = cache[path][key]
            if method == "rules":
                values.update(
                    gap_region=row["gap_region"],
                    rules_reason=row["rule_primary_reason"],
                    rules_decision=row["final_decision"],
                    input_fingerprint=row["input_fingerprint"],
                )
            else:
                values["rf_cnn_decision"] = row["final_label"].upper()
        if values["gap_region"] == "eae_like":
            values.update(
                rules_decision="ROUTED_EAE",
                rf_cnn_decision="ROUTED_EAE",
                rules_reason="frequency_routing",
            )
        return values

    added = current.keys() - previous.keys()
    removed = previous.keys() - current.keys()
    common = current.keys() & previous.keys()
    changed = {
        key
        for key in common
        if any(current[key][f] != previous[key][f] for f in DECISION_FIELDS)
    }
    approved = {
        key
        for key in added | changed
        if key in reviews
        and reviews[key]["input_fingerprint"] == current[key]["input_fingerprint"]
        and reviews[key]["review_decision"] == current[key]["rules_decision"]
    }
    to_review = (added | changed) - approved
    changes = []
    for kind, keys in [("added", added), ("changed", changed), ("removed", removed)]:
        for key in sorted(keys):
            before, after = state(key, "backup"), state(key, "output")
            if key in previous:
                # Older 15-shot comparison uses its v5 verdict, not the v2 backup.
                before.update({f: previous[key][f] for f in DECISION_FIELDS})
            if key in current:
                assert all(after[f] == current[key][f] for f in DECISION_FIELDS), key
            else:
                assert after["rules_decision"] == after["rf_cnn_decision"], key
            status = (
                "already_approved"
                if key in approved
                else ("new_review" if key in to_review else "no_longer_disagrees")
            )
            changes.append(
                dict(
                    change=kind,
                    review_status=status,
                    mode_key=key,
                    in_elena_questions=key in questions,
                    **{"before_" + k: v for k, v in before.items()},
                    **{"after_" + k: v for k, v in after.items()}
                )
            )
    args.out_dir.mkdir(parents=True, exist_ok=True)
    review_rows = [
        dict(path=key, label="", **current[key]) for key in sorted(to_review)
    ]
    write(
        args.out_dir / "to_review.csv",
        ["path", "label", *next(iter(current.values()))],
        review_rows,
    )
    fields = ["change", "review_status", "mode_key", "in_elena_questions"] + [
        prefix + field
        for prefix in ("before_", "after_")
        for field in (
            "gap_region",
            "rules_reason",
            "rules_decision",
            "input_fingerprint",
            "rf_cnn_decision",
        )
    ]
    write(args.out_dir / "changes.csv", fields, changes)
    summary = dict(
        previous_disagreements=len(previous),
        previous_15=len(prior15),
        previous_12=len(previous) - len(prior15),
        current_disagreements=len(current),
        added=len(added),
        removed=len(removed),
        retained=len(common),
        retained_with_changed_decision_reason_or_input=len(changed),
        already_approved_new_disagreements=len(approved),
        new_cases_to_review=len(to_review),
        new_review_by_shot=dict(Counter(k.split("/")[0] for k in sorted(to_review))),
        original_questions=len(questions),
        questions_still_disagree=len(questions.keys() & current.keys()),
        questions_no_longer_disagree=sorted(questions.keys() - current.keys()),
        input_sha256=hashes,
    )
    assert len(previous) - len(removed) + len(added) == len(current)
    assert len(added | changed) == len(approved) + len(to_review)
    for p in inputs.values():
        assert (
            hashlib.sha256(p.read_bytes()).hexdigest()
            == hashes[str(p.relative_to(REPO))]
        )
    (args.out_dir / "summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    print(
        json.dumps({k: v for k, v in summary.items() if k != "input_sha256"}, indent=2)
    )


if __name__ == "__main__":
    main()
