"""Update processed membership after the four verified outputs are installed.

Example: python audits/released4_rules_20261007/record_results.py
"""
from datetime import datetime, timezone
import csv
import json
from pathlib import Path

from run_batch import HERE, REPO, SHOTS, batch


def main():
    inputs = json.loads((HERE / "run_inputs.json").read_text())
    publication = json.loads((HERE / "publication.json").read_text())
    results = json.loads((HERE / "stage_results.json").read_text())
    assert publication["status"] == "installed" and set(publication["installed"]) == set(SHOTS)
    assert not results["failed"] and not (HERE / "inventory_update.json").exists()
    assert all(batch.sha256_file(REPO / p) == h for p, h in inputs["source_sha256"].items())
    summaries = {r["shot"]: r for r in results["verified"]}
    destination = Path(publication["rules_root"])
    for shot in SHOTS:
        assert batch.tree(destination / shot) == summaries[shot]["output_sha256"]
    prepared, changes = {}, []
    for name in ("shot_status.csv", "g_shot_status.csv"):
        path = REPO / "audits/main_dataset_shots" / name
        rows = batch.read(path)
        changed = []
        for row in rows:
            if row["shot"] not in SHOTS:
                continue
            before = dict(row)
            assert row["status"] == "ready_for_rules"
            assert row["post_training_checked"] == row["active_training_shot"] == "no"
            s = summaries[row["shot"]]
            row.update(post_training_checked="yes", checked_methods="rules", status="sorted_rules_pending_review")
            note = (f"2026-10-07: Released-shot production rules v13 sorting completed and verified: "
                    f"{s['input_modes']} inputs, nr=201, {s['bad']} BAD, {s['selected_good']} selected GOOD, "
                    f"{s['eae_like']} EAE-like, zero INVALID. Output: {destination.name}/{row['shot']}. "
                    "Selected GOOD review pending; continuum-release evidence and separate EAE/mixed flags retained. "
                    "See audits/released4_rules_20261007.")
            row["notes"] = row["notes"].rstrip() + " " + note
            changed.append(dict(shot=row["shot"], before=before, after=dict(row)))
        assert len(changed) == 4
        prepared[path] = rows
        changes.append(dict(path=str(path.relative_to(REPO)), before_sha256=batch.sha256_file(path), changes=changed))
    inventory = prepared[batch.INVENTORY]
    byshot = {r["shot"]: r for r in inventory}
    assert all(r == byshot[r["shot"]] for r in prepared[batch.INVENTORY.with_name("g_shot_status.csv")])
    pending = {r["shot"] for r in inventory if r["post_training_checked"] != "yes" and r["active_training_shot"] != "yes"}
    remaining = [r for r in batch.read(REPO / "audits/r42_f62_followup_20261007/remaining_unprocessed.csv")
                 if r["shot"] not in SHOTS]
    assert len(pending) == len(remaining) == 5 and {r["shot"] for r in remaining} == pending
    assert all(r["inventory_status"] == byshot[r["shot"]]["status"] for r in remaining)
    batch.write(HERE / "remaining_unprocessed.csv", remaining)
    assert sum(r["post_training_checked"] == "yes" for r in inventory) == 181
    assert sum(r["active_training_shot"] == "yes" for r in inventory) == 14
    selected_total, pending_good, pending_shots = 0, 0, 0
    for row in inventory:
        if row["post_training_checked"] != "yes":
            continue
        with (destination / row["shot"] / "shot_summary.csv").open() as stream:
            summary = dict(csv.reader(stream))
        count = int(summary["n_final_good"])
        selected_total += count
        if row["status"] == "sorted_rules_pending_review":
            pending_shots += 1
            pending_good += count
    new_good = sum(r["selected_good"] for r in summaries.values())
    assert selected_total == 8318 + new_good
    assert pending_good == 6522 + new_good and pending_shots == 127
    for item in changes:
        path = REPO / item["path"]
        assert batch.sha256_file(path) == item["before_sha256"]
        temporary = path.with_suffix(".csv.tmp")
        batch.write(temporary, prepared[path])
        temporary.replace(path)
        item["after_sha256"] = batch.sha256_file(path)
    reviewed = batch.read(HERE / "reviewed_tae_mode_results.csv")
    batch.save(HERE / "inventory_update.json", dict(created_utc=datetime.now(timezone.utc).isoformat(),
        processed=181, active_training=14, remaining=5, held=4, empty=1,
        selected_good_total=selected_total, pending_good_review_shots=pending_shots,
        pending_good_review_modes=pending_good, new_selected_good=new_good, changes=changes,
        reviewed_mode_final_counts={d: sum(r["final_decision"] == d for r in reviewed) for d in ("GOOD", "BAD")},
        files_sha256={name: batch.sha256_file(HERE / name) for name in
                      ("reviewed_tae_mode_results.csv", "remaining_unprocessed.csv")}))
    print(f"Recorded four processed shots: 181 processed + 14 training; four held + one empty remain.")
    print(f"Selected GOOD total={selected_total}; pending review={pending_good} in {pending_shots} shots.")


if __name__ == "__main__":
    main()
