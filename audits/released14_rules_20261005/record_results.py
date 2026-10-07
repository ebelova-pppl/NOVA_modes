"""Record installed batch outputs and update the two live shot inventories.

Example: python audits/released14_rules_20261005/record_results.py
"""
from datetime import datetime, timezone
import json
from pathlib import Path

from run_batch import HERE, REPO, batch


def main():
    publication = json.loads((HERE / "publication.json").read_text())
    inputs = json.loads((HERE / "run_inputs.json").read_text())
    results = json.loads((HERE / "stage_results.json").read_text())
    assert publication["status"] == "installed" and not results["failed"]
    assert not (HERE / "inventory_update.json").exists()
    selected = set(publication["installed"])
    assert len(selected) == 14 and selected == set(inputs["shots"])
    assert all(batch.sha256_file(REPO / p) == h for p, h in inputs["source_sha256"].items())
    summaries = {r["shot"]: r for r in results["verified"]}
    destination = Path(publication["rules_root"])
    modes = {}
    for shot in sorted(selected):
        assert batch.tree(destination / shot) == summaries[shot]["output_sha256"]
        rows = batch.read(destination / shot / "all_modes_rules.csv")
        modes.update({r["mode_key"]: r for r in rows})
    # Record actual gate outcomes without inventing individual manual labels
    # from the user's aggregate morphology review.
    reviewed = []
    for old in batch.read(REPO / "audits/continuum_small_review_20261005/tae_only_small_count_modes.csv"):
        r = modes[old["path"]]
        assert r["input_fingerprint"] == old["input_fingerprint"]
        row = {k: r.get(k, "") for k in ("mode_key", "shot", "ntor", "gap_region", "rule_decision",
               "rule_primary_reason", "final_decision", "selected_final", "overall_rule_severity", "input_fingerprint")}
        row["path"] = r["mode_key"]
        row["user_review_note"] = ("Presentable; user sees no continuum-crossing issue" if
            r["mode_key"] == "nstxuG142301E34/N2/egn02w.2204E+02" else
            "Included in aggregate morphology review; no individual override assigned")
        reviewed.append(row)
    assert len(reviewed) == 39
    batch.write(HERE / "reviewed_tae_mode_results.csv", reviewed)
    changes = []
    prepared = {}
    for name in ("shot_status.csv", "g_shot_status.csv"):
        path = REPO / "audits/main_dataset_shots" / name
        rows = batch.read(path)
        changed = []
        for row in rows:
            if row["shot"] not in selected:
                continue
            before = dict(row)
            assert row["status"] == "ready_for_rules"
            assert row["post_training_checked"] == row["active_training_shot"] == "no"
            s = summaries[row["shot"]]
            row.update(post_training_checked="yes", checked_methods="rules", status="sorted_rules_pending_review")
            note = (f"2026-10-05: Released-shot production rules v13 sorting completed and verified: "
                    f"{s['input_modes']} inputs, nr=201, {s['bad']} BAD, {s['selected_good']} selected GOOD, "
                    f"{s['eae_like']} EAE-like, zero INVALID. Output: {destination.name}/{row['shot']}. "
                    "Full selected-mode review pending; separate EAE/mixed evidence retained. "
                    "See audits/released14_rules_20261005.")
            row["notes"] = row["notes"].rstrip() + " " + note
            changed.append(dict(shot=row["shot"], before=before, after=dict(row)))
        assert len(changed) == 14
        prepared[path] = rows
        changes.append(dict(path=str(path.relative_to(REPO)), before_sha256=batch.sha256_file(path), changes=changed))
    inventory = prepared[REPO / "audits/main_dataset_shots/shot_status.csv"]
    byshot = {r["shot"]: r for r in inventory}
    assert all(r == byshot[r["shot"]] for r in prepared[REPO / "audits/main_dataset_shots/g_shot_status.csv"])
    pending = [r for r in inventory if r["post_training_checked"] != "yes" and r["active_training_shot"] != "yes"]
    holds = {r["shot"]: r for r in batch.read(REPO / "audits/continuum_release_20261005/remaining_continuum_holds.csv")}
    assert len(pending) == 9 and {r["shot"] for r in pending} == set(holds) | {"nstxu_202806"}
    remaining = []
    for r in pending:
        h = holds.get(r["shot"], {})
        remaining.append(dict(shot=r["shot"], reason="TAE crossing correspondence review pending" if h else "Empty entry; no active egn inputs",
            n1_flagged_tae_comparisons=h.get("tae_like_n1_flagged_comparisons", ""),
            n2_flagged_tae_comparisons=h.get("tae_like_n2_flagged_comparisons", ""),
            total_flagged_tae_comparisons=h.get("tae_like_flagged_comparisons", ""),
            potential_eae_issue=h.get("eae_review_status", "")))
    batch.write(HERE / "remaining_unprocessed.csv", remaining)
    assert sum(r["post_training_checked"] == "yes" for r in inventory) == 177
    assert sum(r["active_training_shot"] == "yes" for r in inventory) == 14
    for c in changes:
        path = REPO / c["path"]
        assert batch.sha256_file(path) == c["before_sha256"]
        temporary = path.with_suffix(".csv.tmp")
        batch.write(temporary, prepared[path])
        temporary.replace(path)
        c["after_sha256"] = batch.sha256_file(path)
    batch.save(HERE / "inventory_update.json", dict(created_utc=datetime.now(timezone.utc).isoformat(),
        processed=177, active_training=14, remaining=9, continuum_holds=8, empty_entries=1, changes=changes,
        files_sha256={name: batch.sha256_file(HERE / name) for name in
                      ("reviewed_tae_mode_results.csv", "remaining_unprocessed.csv")}))
    print("Recorded 14 processed shots: 177 processed + 14 training, eight holds + one empty remain.")
    print("Reviewed-mode outcomes:", {d: sum(r["final_decision"] == d for r in reviewed) for d in ("GOOD", "BAD")})


if __name__ == "__main__":
    main()
