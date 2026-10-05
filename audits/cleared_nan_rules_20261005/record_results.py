"""Record verified installed outputs and list shots with no completed processing.

Run after install.py: python audits/cleared_nan_rules_20261005/record_results.py
"""
import csv
from datetime import datetime, timezone
import io
import json
from pathlib import Path

from run_batch import HERE, REPO, batch


def main():
    publication = json.loads((HERE / "publication.json").read_text())
    inputs = json.loads((HERE / "run_inputs.json").read_text())
    results = json.loads((HERE / "stage_results.json").read_text())
    assert publication["status"] == "installed"
    assert not (HERE / "inventory_update.json").exists()
    selected = set(publication["installed"])
    assert selected == {r["shot"] for r in batch.read(HERE / "selection.csv")}
    assert len(selected) == 5 and not results["failed"]
    assert all(batch.sha256_file(REPO / p) == h for p, h in inputs["protected_sha256"].items())
    summaries = {r["shot"]: r for r in results["verified"]}
    candidates = []
    destination = Path(publication["destination_root"])
    for shot in sorted(selected):
        assert batch.tree(destination / shot) == summaries[shot]["output_sha256"]
        filename = "review_tae_like.csv" if inputs["workflow"] == "rules-cli" else "good_tae_final.csv"
        for r in batch.read(destination / shot / filename):
            fields = ["path", "mode_key", "shot", "n", "ntor", "nr", "nhar", "omega", "gamma_d",
                      "rad_loc", "rad_width", "gap_region", "rule_decision", "rule_primary_reason",
                      "final_decision", "decision_source", "selected_final", "overall_rule_severity", "nearest_gate", "input_fingerprint"]
            c = {key: r.get(key, "") for key in fields}
            c["path"] = r["mode_key"]
            candidates.append(c)
    candidate_file = "review_candidates.csv" if inputs["workflow"] == "rules-cli" else "good_tae_final_batch.csv"
    batch.write(HERE / candidate_file, candidates)
    changes = []
    for filename in ("shot_status.csv", "g_shot_status.csv"):
        path = REPO / "audits/main_dataset_shots" / filename
        before_hash = batch.sha256_file(path)
        reader = csv.DictReader(io.StringIO(path.read_text()))
        fields, rows = reader.fieldnames, list(reader)
        changed = []
        for row in rows:
            if row["shot"] not in selected:
                continue
            before = dict(row)
            assert row["post_training_checked"] == row["active_training_shot"] == "no"
            s = summaries[row["shot"]]
            calibration = inputs["workflow"] == "rules-cli"
            row.update(post_training_checked="yes", checked_methods="rules-calibration" if calibration else "rules",
                       status="sorted_rules_calibration_pending_review" if calibration else "sorted_rules_pending_review")
            note = (f"2026-10-05: NaN hold resolved; user-requested {inputs['workflow']} v13 sorting completed and verified. "
                    f"{s['input_modes']} inputs, nr=201, {s['bad']} BAD, {s['review']} REVIEW, "
                    f"{s['selected_good']} selected GOOD, {s['eae_like']} EAE-like, zero INVALID. "
                    f"Output: {destination.name}/{row['shot']}. Visual review pending; "
                    "this run does not declare N1/N2 alignment cleared. See audits/cleared_nan_rules_20261005.")
            if row["shot"] == "nstxuG142301M21":
                note += " Existing N1/N2 log-coverage limitations remain recorded."
            if row["shot"] == "nstxuE203655F01t030":
                note += " Existing isolated N2/6049 crossing-correspondence question remains recorded."
            row["notes"] = (row["notes"].rstrip() + " " + note).strip()
            changed.append(dict(shot=row["shot"], before=before, after=dict(row)))
        assert len(changed) == (5 if filename == "shot_status.csv" else 1)
        temporary = path.with_suffix(".csv.tmp")
        batch.write(temporary, rows, fields)
        temporary.replace(path)
        changes.append(dict(path=str(path.relative_to(REPO)), before_sha256=before_hash,
                            after_sha256=batch.sha256_file(path), changes=changed))
    inventory = batch.read(REPO / "audits/main_dataset_shots/shot_status.csv")
    pending = [r for r in inventory if r["post_training_checked"] != "yes" and r["active_training_shot"] != "yes"]
    n1 = {r["shot"]: r for r in batch.read(REPO / "audits/n1_recheck_20261005/n1_before_after.csv")}
    n2 = {r["shot"]: r for r in batch.read(REPO / "audits/n1_recheck_20261005/n2_control_before_after.csv")}
    remaining = []
    for item in pending:
        shot = item["shot"]
        row = dict(shot=shot, reason="", n1_tae_beyond_2="", n1_tae_comparisons="",
                   n1_all_beyond_2="", n1_all_comparisons="", n2_tae_beyond_2="", n2_tae_comparisons="",
                   n2_all_beyond_2="", n2_all_comparisons="", notes="")
        if shot in n1:
            for n, table in ((1, n1), (2, n2)):
                for cohort, target in (("tae_side", "tae"), ("all", "all")):
                    row[f"n{n}_{target}_beyond_2"] = table[shot][f"current_{cohort}_beyond_2"]
                    row[f"n{n}_{target}_comparisons"] = table[shot][f"current_{cohort}_crossings"]
            row["reason"] = "N1/N2 continuum correspondence still requires review"
            if int(n1[shot]["current_tae_side_beyond_2"]) == 0:
                row["reason"] = "N1 TAE-side screen passes; residual N2/all-frequency correspondence requires review"
            if shot == "nstxuG142301L89":
                row["notes"] = "Marginal: one N1 EAE-side offset 2.16 and one N2 TAE-side offset 2.35 grid intervals; not a confirmed whole-shot failure."
            if shot == "nstxuG142301D46":
                row["notes"] = "NaN gamma_d issue resolved; separate continuum-review hold remains."
            if shot == "nstxuG121123Q62":
                row["notes"] = "Training remains suspended separately."
            if shot in ("nstxuG133964R48", "nstxuG133964U27"):
                row["notes"] = "Unchanged secondary-review case; not confirmed invalid."
        else:
            assert shot == "nstxu_202806"
            row["reason"] = "Empty database entry; no active egn mode inputs"
        remaining.append(row)
    assert len(pending) == len(remaining) == 23 and len(set(n1) & {r['shot'] for r in remaining}) == 22
    batch.write(HERE / "remaining_unprocessed.csv", remaining)
    lines = ["Remaining unprocessed shots (2026-10-05): 23", "", "Continuum correspondence review: 22"]
    lines += [f"{r['shot']}: {r['reason']}" + (f" ({r['notes']})" if r['notes'] else "") for r in remaining if r['shot'] in n1]
    lines += ["", "Empty entry: nstxu_202806", "", (
        "Five newly processed shots have REVIEW survivors from sort_shot_rules.py and still need review/production promotion."
        if inputs["workflow"] == "rules-cli" else
        "Five newly processed shots have production GOOD lists; visual review is pending.")]
    (HERE / "remaining_unprocessed.txt").write_text("\n".join(lines) + "\n")
    assert sum(r["post_training_checked"] == "yes" for r in inventory) == 163
    assert sum(r["active_training_shot"] == "yes" for r in inventory) == 14
    assert len(candidates) == sum(s["review"] if inputs["workflow"] == "rules-cli" else s["selected_good"] for s in summaries.values())
    batch.save(HERE / "inventory_update.json", dict(created_utc=datetime.now(timezone.utc).isoformat(),
               processed=163, production_processed=158 if inputs["workflow"] == "rules-cli" else 163,
               calibration_processed=5 if inputs["workflow"] == "rules-cli" else 0,
               active_training=14, remaining=23, changes=changes))
    print(f"Recorded five processed shots, {len(candidates)} candidates, and 23 remaining entries.")


if __name__ == "__main__":
    main()
