"""Selection and provenance safeguards for the shared accepted-mode catalog."""

from pathlib import Path
import sys
import tempfile
import unittest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
from build_accepted_catalog import FIELDS, build_catalog, compact_row, read_reviews
from input_validity import FIELDS as REGISTRY_FIELDS, InputValidityRegistry
from tae_rule_io import write_dict_csv


SHOT = "nstxu_204202"
KEY = SHOT + "/N4/egn04w.8263E+01"
FINGERPRINT = "a" * 64


def sample():
    return dict(path="/example/DiTw/" + KEY, mode_key=KEY, shot=SHOT, ntor="4",
                omega="2.8744578114322614", rad_loc="0.5558928093579375",
                rad_width="0.12265701830453488", gap_region="tae_like",
                input_fingerprint=FINGERPRINT, rule_version="old-engine",
                manual_decision="", override_status="NOT_PRESENT",
                decision_source="rule_survivor_policy", rule_decision="REVIEW",
                final_decision="GOOD", selected_final="True")


SUMMARY = dict(shot=SHOT, n_final_good="1", rule_configuration_name="tae_rules_production_v1")
REGISTRY = InputValidityRegistry("", {})


class AcceptedCatalogTests(unittest.TestCase):
    def compact(self, row=None, status="training", reviews=None, registry=REGISTRY):
        return compact_row(row or sample(), SHOT, SUMMARY, status, reviews or {}, registry)

    def test_preliminary_review_is_exported_as_final_good_with_legacy_blanks(self):
        row = self.compact()
        self.assertEqual(set(row), set(FIELDS))
        self.assertEqual(len(row), 12)
        self.assertEqual(row["path"], KEY)
        self.assertEqual(row["final_decision"], "GOOD")
        self.assertEqual(row["review_status"], "unknown")
        self.assertEqual(row["overall_rule_severity"], "")
        self.assertEqual(row["omega"], sample()["omega"])
        self.assertEqual(row["rad_loc"], "0.5559")

    def test_only_matching_mode_and_continuum_fingerprint_marks_reviewed(self):
        stale = {(KEY, "b" * 64): [dict(source="review.csv", decision="GOOD")]}
        matching = {(KEY, FINGERPRINT): [dict(source="review.csv", decision="GOOD")]}
        self.assertEqual(self.compact(reviews=stale)["review_status"], "unknown")
        self.assertEqual(self.compact(reviews=matching)["review_status"], "reviewed")
        self.assertEqual(self.compact(status="sorted_rules_pending_review")["review_status"], "pending")

    def test_manual_rescue_keeps_above_threshold_severity(self):
        row = sample()
        row.update(rule_decision="BAD", manual_decision="GOOD", override_status="APPLIED",
                   decision_source="manual_override", overall_rule_severity="1.420324614")
        result = self.compact(row)
        self.assertEqual(result["manual_override"], "yes")
        self.assertEqual(result["overall_rule_severity"], "1.42")
        self.assertEqual(result["review_status"], "reviewed")

    def test_rejects_nonselected_bad_inconsistent_or_nonfinite_rows(self):
        for change in ({"selected_final": "False"}, {"final_decision": "BAD"},
                       {"ntor": "6"}, {"shot": "different"}, {"rad_loc": "nan"},
                       {"overall_rule_severity": "inf"}, {"manual_decision": "GOOD"},
                       {"rule_configuration_name": "different-preset"}):
            with self.subTest(change=change), self.assertRaises(ValueError):
                self.compact({**sample(), **change})

    def test_known_invalid_scope_is_not_silently_republished(self):
        issue = dict(issue_id="TEST", reason="invalid", reviewer="tester", review_date="today", evidence="test")
        registry = InputValidityRegistry("hash", {(SHOT, 4): issue})
        with self.assertRaisesRegex(ValueError, "excluded input"):
            self.compact(registry=registry)

    def test_review_registry_respects_shot_scope_and_preserves_disagreements(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            evidence = [dict(path=KEY, after="GOOD", input_fingerprint=FINGERPRINT),
                        dict(path="different/N4/egn04w.1", after="GOOD", input_fingerprint=FINGERPRINT),
                        dict(path=SHOT + "/N4/egn04w.2", after="BAD", input_fingerprint=FINGERPRINT)]
            write_dict_csv(root / "review.csv", evidence[0].keys(), evidence)
            specs = [dict(path="review.csv", decision_column="after", shot=SHOT, reviewer="Elena")]
            write_dict_csv(root / "sources.csv", specs[0].keys(), specs)
            reviews, _ = read_reviews(root / "sources.csv", repo=root)
            self.assertEqual(set(reviews), {(KEY, FINGERPRINT), (SHOT + "/N4/egn04w.2", FINGERPRINT)})

    def test_training_human_bad_vs_rules_good_needs_adjudication(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            labels = [dict(path=KEY, validity="bad")]
            fingerprints = [dict(mode_key=KEY, input_fingerprint=FINGERPRINT)]
            specs = [dict(path="labels.csv", decision_column="validity", shot="*", reviewer="Elena", fingerprint_source="fingerprints.csv")]
            for name, rows in (("labels.csv", labels), ("fingerprints.csv", fingerprints), ("sources.csv", specs)):
                write_dict_csv(root / name, rows[0].keys(), rows)
            reviews, _ = read_reviews(root / "sources.csv", repo=root)
            result = self.compact(reviews=reviews)
            self.assertEqual(result["final_decision"], "GOOD")
            self.assertEqual(result["review_status"], "pending")

    def test_complete_inventory_and_exact_membership_required(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            output = root / "sort" / SHOT
            output.mkdir(parents=True)
            inventory = [dict(shot=SHOT, status="training", active_training_shot="yes", post_training_checked="no")]
            write_dict_csv(root / "inventory.csv", inventory[0].keys(), inventory)
            write_dict_csv(root / "reviews.csv", ("path", "decision_column", "shot", "reviewer"), [])
            write_dict_csv(root / "invalid.csv", REGISTRY_FIELDS, [])
            write_dict_csv(output / "shot_summary_wide.csv", SUMMARY.keys(), [SUMMARY])
            args = (root / "sort", root / "inventory.csv", root / "reviews.csv", root / "invalid.csv")
            with self.assertRaisesRegex(ValueError, "Missing final lists"):
                build_catalog(*args)
            row = sample()
            write_dict_csv(output / "good_tae_final.csv", row.keys(), [row])
            # A nested backup is deliberately not a second catalog source.
            write_dict_csv(root / "sort/backup" / SHOT / "good_tae_final.csv", row.keys(), [row])
            rows, metadata = build_catalog(*args)
            self.assertEqual([r["path"] for r in rows], [KEY])
            self.assertEqual(metadata["total_modes"], 1)
            write_dict_csv(output / "good_tae_final.csv", row.keys(), [row, row])
            with self.assertRaisesRegex(ValueError, "row count"):
                build_catalog(*args)
            write_dict_csv(output / "shot_summary_wide.csv", SUMMARY.keys(), [{**SUMMARY, "n_final_good": "2"}])
            with self.assertRaisesRegex(ValueError, "Duplicate selected mode"):
                build_catalog(*args)


if __name__ == "__main__":
    unittest.main()
