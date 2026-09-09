"""Record changes from the preceding 27-shot disagreement list.

Run after adopt.py stage/verify:
  python audits/cross_window_exception_20260909/compare_disagreements.py
"""

from collections import Counter
import csv
import hashlib
import json
from pathlib import Path

HERE = Path(__file__).resolve().parent


def read(path):
    with path.open() as handle:
        return {row["mode_key"]: row for row in csv.DictReader(handle)}


def main():
    paths = {
        "previous": HERE.parent
        / "continuum_monotonic_tail_20260908/regenerated_disagreements.csv",
        "current": HERE / "regenerated_disagreements.csv",
        "review": HERE / "user_review.csv",
        "adopted": HERE / "adopted_changes.csv",
    }
    old, new = read(paths["previous"]), read(paths["current"])
    approved, adopted = read(paths["review"]), read(paths["adopted"])
    for key in old.keys() & new.keys():
        assert old[key] == new[key], key
    changes = []
    fields = [
        "mode_key",
        "shot",
        "input_fingerprint",
        "change",
        "rules_before",
        "rules_after",
        "rf_cnn_decision",
        "review_decision",
        "needs_new_review",
    ]
    for key in sorted(old.keys() ^ new.keys()):
        added = key in new
        row = new[key] if added else old[key]
        assert (
            key in adopted
            and row["input_fingerprint"] == adopted[key]["input_fingerprint"]
        )
        reviewed = (
            key in approved
            and row["input_fingerprint"] == approved[key]["input_fingerprint"]
            and approved[key]["review_decision"] == "GOOD"
        )
        changes.append(
            dict(
                mode_key=key,
                shot=row["shot"],
                input_fingerprint=row["input_fingerprint"],
                change="added" if added else "removed",
                rules_before="BAD",
                rules_after="GOOD",
                rf_cnn_decision=row["rf_cnn_decision"],
                review_decision="GOOD" if reviewed else "",
                needs_new_review="yes" if added and not reviewed else "no",
            )
        )
    assert len(old) == 229 and len(new) == 234
    assert Counter(r["change"] for r in changes) == {"added": 12, "removed": 7}
    assert not any(r["needs_new_review"] == "yes" for r in changes)
    with (HERE / "disagreement_changes.csv").open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields, lineterminator="\n")
        writer.writeheader()
        writer.writerows(changes)
    summary = dict(
        previous=len(old),
        current=len(new),
        retained=len(old.keys() & new.keys()),
        added=12,
        removed=7,
        new_unreviewed=0,
        current_directions=dict(
            Counter(
                f'{r["rules_decision"]}/{r["rf_cnn_decision"]}' for r in new.values()
            )
        ),
        source_sha256={
            str(p): hashlib.sha256(p.read_bytes()).hexdigest() for p in paths.values()
        },
    )
    (HERE / "disagreement_delta.json").write_text(json.dumps(summary, indent=2) + "\n")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
