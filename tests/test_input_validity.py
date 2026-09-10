"""Known-invalid input coverage before either routing or classification."""

import contextlib
import csv
import io
from pathlib import Path
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import Mock, patch

from test_rule_sorting import write_mode, write_datcon
from input_validity import (
    REGISTRY_PATH,
    KNOWN_INVALID_INPUT,
    load_input_validity_registry,
)
from make_tae_like_list import preprocess_shot
from sort_shot_rules import apply_manual_overrides, run_configured_shot
from sort_shot_mixed import parse_args, run_rf_cnn_method


SHOT = "nstxuG142301C50"
WHOLE_SHOT = "nstxuG133964R06"


def fixture(root):
    shot = root / SHOT
    for n, name, omega in [(1, "tae", 1.0), (1, "eae", 3.0), (2, "control", 1.0)]:
        write_mode(shot / f"N{n}" / f"egn0{n}w.{name}", omega=omega, ntor=n, nr=201)
        write_datcon(shot / f"N{n}" / f"datcon{n}", nr=201)
    return shot


class InputValidityTests(unittest.TestCase):
    def test_registry_exact_scope_and_malformed_policy_fail(self):
        registry = load_input_validity_registry()
        self.assertIn("CONTINUUM_MODE_MISMATCH", registry.diagnostic(SHOT, 1))
        self.assertIsNone(registry.diagnostic(SHOT, 2))
        self.assertIsNone(registry.diagnostic(SHOT + "_new", 1))
        for confirmed in (
            "nstx_135388", "nstxuG142301W29", "nstxuG142301Y93", "nstxuG121123B12"
        ):
            self.assertIn("CONTINUUM_MODE_MISMATCH", registry.diagnostic(confirmed, 1))
            self.assertIsNone(registry.diagnostic(confirmed, 2))
            self.assertIsNone(registry.diagnostic(confirmed + "_new", 1))
        for n in (1, 10, 11):
            self.assertIn(
                "SUSPECT_EIGENMODE_STRUCTURE", registry.diagnostic(WHOLE_SHOT, n)
            )
        self.assertIsNone(registry.diagnostic(WHOLE_SHOT + "_new", 1))
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "registry.csv"
            original = REGISTRY_PATH.read_text()
            for text in [
                "wrong,header\n",
                original + original.splitlines()[1] + "\n",
                original.replace(",1,CONTINUUM", ",0,CONTINUUM"),
                original.replace(",user,", ",,"),
                original.replace(",*,", ",all,"),
                original + original.splitlines()[2] + "\n",
            ]:
                path.write_text(text)
                with self.assertRaises(ValueError):
                    load_input_validity_registry(path)
            # A per-n entry cannot narrow an existing whole-shot exclusion.
            path.write_text(
                original
                + original.splitlines()[2]
                .replace(",*,", ",1,")
                .replace(",SUSPECT_EIGENMODE_STRUCTURE,", ",SPECIFIC_ISSUE,")
                + "\n"
            )
            combined = load_input_validity_registry(path)
            self.assertIn(
                "SUSPECT_EIGENMODE_STRUCTURE", combined.diagnostic(WHOLE_SHOT, 1)
            )

    def test_whole_shot_excludes_every_n_in_both_sorters(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            shot = root / WHOLE_SHOT
            fixture(root).rename(shot)
            # The policy also covers a toroidal n absent from today's data.
            write_mode(shot / "N11/egn11w.future", omega=1.0, ntor=11, nr=201)
            write_datcon(shot / "N11/datcon11", nr=201)
            result = run_configured_shot(
                shot, root / "rules", rule_config="tae_rules_production_v7", n_max=11
            )
            self.assertEqual(len(result.final_rows), 4)
            self.assertTrue(
                all(r["final_decision"] == "INVALID" for r in result.final_rows)
            )
            self.assertEqual(result.summary["n_known_invalid_inputs"], 4)
            self.assertEqual(result.summary["n_rule_evaluated"], 0)
            self.assertEqual(result.summary["n_final_good"], 0)
            out = root / "ai"
            cnn = SimpleNamespace(predict=Mock())
            module = SimpleNamespace(load_cnn_classifier=Mock(return_value=cnn))
            args = parse_args(
                [
                    "--method",
                    "rf-cnn",
                    "--shot_dir",
                    str(shot),
                    "--out_dir",
                    str(out),
                    "--rf_model",
                    "dummy_rf",
                    "--cnn_model",
                    "dummy_cnn",
                    "--device",
                    "cpu",
                    "--n_max",
                    "11",
                ]
            )
            with patch.dict("sys.modules", {"cnn_infer_common": module}), patch(
                "joblib.load", return_value=object()
            ), patch("sort_shot.classify_mode_rf") as rf, contextlib.redirect_stdout(
                io.StringIO()
            ):
                run_rf_cnn_method(args)
            rf.assert_not_called()
            cnn.predict.assert_not_called()
            with (out / "all_modes_scored.csv").open() as handle:
                rows = list(csv.DictReader(handle))
            self.assertEqual(len(rows), 4)
            for row in rows:
                self.assertEqual(row["final_label"], "invalid")
                self.assertEqual(row["rejection_reason"], KNOWN_INVALID_INPUT)
                self.assertEqual(row["gap_region"], "")
                self.assertEqual(row["p_rf_good"], "")
                self.assertEqual(row["p_cnn_good"], "")
            with (out / "shot_summary_wide.csv").open() as handle:
                summary = next(csv.DictReader(handle))
            self.assertEqual(summary["n_known_invalid_inputs"], "4")
            self.assertEqual(summary["n_sent_to_classifiers"], "0")
            self.assertEqual(summary["n_final_good"], "0")

    def test_rules_invalidates_tae_and_eae_before_routing_and_overrides(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            shot = fixture(root)
            original = {p: p.read_bytes() for p in shot.glob("N*/*")}
            result = run_configured_shot(
                shot, root / "rules", rule_config="tae_rules_production_v7"
            )
            rejected = [r for r in result.final_rows if r["ntor"] == 1]
            self.assertEqual(len(rejected), 2)
            for row in rejected:
                self.assertEqual(row["final_decision"], "INVALID")
                self.assertEqual(
                    row["preprocessing_primary_reason"], KNOWN_INVALID_INPUT
                )
                self.assertEqual(row["gap_region"], "")
                self.assertEqual(row["rule_version"], "")
                self.assertEqual(len(row["input_fingerprint"]), 64)
                override = dict(
                    mode_key=row["mode_key"],
                    input_fingerprint=row["input_fingerprint"],
                    manual_decision="GOOD",
                    manual_reason="cannot rescue invalid data",
                )
                applied, audit = apply_manual_overrides([row], [override])
                self.assertEqual(applied[0]["final_decision"], "INVALID")
                self.assertEqual(audit.ineligible, 1)
            self.assertEqual(result.summary["n_known_invalid_inputs"], 2)
            self.assertEqual(result.summary["n_rule_evaluated"], 1)
            for path, data in original.items():
                self.assertEqual(path.read_bytes(), data)
            # Same file contents under an unrelated shot still route normally.
            other = root / "other_shot"
            shot.rename(other)
            control = preprocess_shot(other)
            self.assertEqual(len(control.invalid_rows), 0)
            self.assertEqual(len(control.eae_rows), 1)
            self.assertEqual(len(control.tae_rows), 2)

    def test_rf_cnn_excludes_both_families_before_inference(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            shot = fixture(root)
            out = root / "ai"
            cnn = SimpleNamespace(predict=Mock(return_value={"p_good": 0.1}))
            module = SimpleNamespace(load_cnn_classifier=Mock(return_value=cnn))
            args = parse_args(
                [
                    "--method",
                    "rf-cnn",
                    "--shot_dir",
                    str(shot),
                    "--out_dir",
                    str(out),
                    "--rf_model",
                    "dummy_rf",
                    "--cnn_model",
                    "dummy_cnn",
                    "--device",
                    "cpu",
                ]
            )
            with patch.dict("sys.modules", {"cnn_infer_common": module}), patch(
                "joblib.load", return_value=object()
            ), patch(
                "sort_shot.classify_mode_rf", return_value=(0.1, None, 1.0, 0.01, 2)
            ) as rf, contextlib.redirect_stdout(
                io.StringIO()
            ):
                run_rf_cnn_method(args)
            self.assertEqual(rf.call_count, 1)
            self.assertTrue(rf.call_args.args[1].endswith("N2/egn02w.control"))
            cnn.predict.assert_called_once_with(str(shot / "N2/egn02w.control"))
            with (out / "all_modes_scored.csv").open() as handle:
                rows = list(csv.DictReader(handle))
            rejected = [r for r in rows if r["ntor"] == "1"]
            self.assertEqual(len(rejected), 2)
            for row in rejected:
                self.assertEqual(row["final_label"], "invalid")
                self.assertEqual(row["status"], "rejected")
                self.assertEqual(row["rejection_reason"], KNOWN_INVALID_INPUT)
                self.assertEqual(row["p_rf_good"], "")
                self.assertEqual(row["p_cnn_good"], "")
                self.assertEqual(row["gap_region"], "")
            with (out / "shot_summary_wide.csv").open() as handle:
                summary = next(csv.DictReader(handle))
            self.assertEqual(summary["n_known_invalid_inputs"], "2")
            self.assertEqual(summary["n_nan_or_invalid"], "2")
            self.assertEqual(summary["n_sent_to_classifiers"], "1")


if __name__ == "__main__":
    unittest.main()
