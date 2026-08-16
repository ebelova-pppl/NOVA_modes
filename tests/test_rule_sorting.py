import csv
import json
import os
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path

import numpy as np


REPO_ROOT = Path(__file__).resolve().parents[1]
SRC_DIR = REPO_ROOT / "src"
SCRIPTS_DIR = REPO_ROOT / "scripts"
for directory in (SRC_DIR, SCRIPTS_DIR):
    if str(directory) not in sys.path:
        sys.path.insert(0, str(directory))

from label_modes_fast import adjudication_candidate_rows  # noqa: E402
from make_tae_like_list import (  # noqa: E402
    classify_gap_region,
    preprocess_shot,
)
from nova_mode_loader import load_mode_from_nova  # noqa: E402
from sort_shot_mixed import classify_gap_region as mixed_classify_gap_region  # noqa: E402
from sort_shot_rules import (  # noqa: E402
    OverrideAudit,
    apply_manual_overrides,
    build_summary,
    deduplicate_final_good,
    load_manual_overrides,
    run_shot,
)
from tae_rule_engine import RULESET_NOT_IMPLEMENTED, evaluate_mode  # noqa: E402
from tae_rule_io import (  # noqa: E402
    MANUAL_OVERRIDE_FIELDS,
    RULE_OUTPUT_FIELDS,
    input_fingerprint,
    portable_mode_key,
    read_dict_csv,
    sha256_file,
    stable_json,
    write_dict_csv,
)


REQUIRED_OUTPUTS = {
    "all_modes_rules.csv",
    "tae_like_all.csv",
    "eae_like.csv",
    "rejected_modes.csv",
    "rule_results.csv",
    "final_classifications.csv",
    "bad_tae_like.csv",
    "review_tae_like.csv",
    "good_tae_unchecked.csv",
    "good_tae_final.csv",
    "manual_overrides.csv",
    "shot_summary.csv",
    "shot_summary_wide.csv",
    "shot_summary_by_n.csv",
    "frequency_cluster_report.txt",
    "frequency_clusters.csv",
}


def write_mode(
    path: Path,
    *,
    omega: float,
    ntor: int,
    nr: int = 10,
    mode: np.ndarray | None = None,
) -> np.ndarray:
    nhar = 4 * ntor
    if mode is None:
        r = np.linspace(0.0, 1.0, nr)
        envelope = np.exp(-((r - 0.4) / 0.18) ** 2)
        mode = np.stack([(index + 1) * envelope for index in range(nhar)])
    payload = np.zeros((3, nhar, nr), dtype=float)
    payload[0] = mode
    values = np.concatenate(
        (
            np.array([omega], dtype=float),
            payload.reshape(-1),
            np.array([nr, 0.01, ntor], dtype=float),
        )
    )
    path.parent.mkdir(parents=True, exist_ok=True)
    values.tofile(path)
    return mode


def write_datcon(
    path: Path,
    *,
    nr: int = 10,
    lower_frequency: float | np.ndarray = 0.5,
    upper_frequency: float | np.ndarray = 1.5,
) -> None:
    lower = np.broadcast_to(np.asarray(lower_frequency, dtype=float), (nr,))
    upper = np.broadcast_to(np.asarray(upper_frequency, dtype=float), (nr,))
    lines = [f"1 {nr}"]
    lines.extend(f"{low**2:.17g} {high**2:.17g}" for low, high in zip(lower, upper))
    path.write_text("\n".join(lines) + "\n")


def make_tae_shot(root: Path, *, names: tuple[str, ...] = ("egn01w.one",)) -> Path:
    shot = root / "synthetic_shot"
    n_dir = shot / "N1"
    n_dir.mkdir(parents=True)
    write_datcon(n_dir / "datcon1")
    for index, name in enumerate(names):
        write_mode(n_dir / name, omega=1.0 + 0.005 * index, ntor=1)
    return shot


def override_row(result_row: dict[str, str], decision: str = "GOOD") -> dict[str, str]:
    return {
        "mode_key": result_row["mode_key"],
        "path": result_row["path"],
        "input_fingerprint": result_row["input_fingerprint"],
        "ntor": result_row["ntor"],
        "frequency": result_row["omega"],
        "original_rule_decision": result_row["rule_decision"],
        "manual_decision": decision,
        "manual_reason": "explicit synthetic adjudication",
        "reviewer": "unit-test",
        "adjudication_timestamp": "2026-08-15T00:00:00Z",
    }


class PreprocessingTests(unittest.TestCase):
    def test_missing_datcon_aborts_before_mode_processing(self):
        with tempfile.TemporaryDirectory() as temporary:
            shot = Path(temporary) / "shot_missing_datcon"
            n_dir = shot / "N1"
            n_dir.mkdir(parents=True)
            (n_dir / "egn01w.broken").write_bytes(b"not a NOVA mode")
            with self.assertRaisesRegex(SystemExit, "required continuum file is missing"):
                preprocess_shot(shot)

    def test_malformed_mode_is_invalid_while_valid_mode_continues(self):
        with tempfile.TemporaryDirectory() as temporary:
            shot = make_tae_shot(Path(temporary))
            (shot / "N1" / "egn01w.broken").write_bytes(b"broken")
            result = preprocess_shot(shot)

        self.assertEqual(len(result.rows), 2)
        self.assertEqual(len(result.tae_rows), 1)
        self.assertEqual(len(result.invalid_rows), 1)
        self.assertEqual(
            result.invalid_rows[0]["preprocessing_primary_reason"], "MODE_LOAD_FAILED"
        )

    def test_tae_eae_mixed_routing_matches_canonical_function(self):
        cases = [(0.3, 0.6), (-0.8, 0.2), (0.2, 0.45)]
        for signed_delta, fraction in cases:
            expected = mixed_classify_gap_region(
                signed_delta,
                fraction,
                fraction_tae_threshold=0.5,
                fraction_eae_threshold=0.4,
                signed_delta_eae_threshold=-0.1,
            )
            actual = classify_gap_region(signed_delta, fraction)
            self.assertEqual(actual, expected)

    def test_input_fingerprint_changes_for_either_file(self):
        with tempfile.TemporaryDirectory() as temporary:
            shot = make_tae_shot(Path(temporary))
            mode = shot / "N1" / "egn01w.one"
            datcon = shot / "N1" / "datcon1"
            original = input_fingerprint(mode, datcon)
            mode.write_bytes(mode.read_bytes() + b"x")
            changed_mode = input_fingerprint(mode, datcon)
            mode.write_bytes(mode.read_bytes()[:-1])
            datcon.write_text(datcon.read_text() + "\n")
            changed_datcon = input_fingerprint(mode, datcon)

        self.assertNotEqual(original, changed_mode)
        self.assertNotEqual(original, changed_datcon)


class RuleAndOverrideTests(unittest.TestCase):
    def setUp(self):
        self.base = {
            "path": "/data/shot/N1/egn01w.one",
            "mode_key": "shot/N1/egn01w.one",
            "shot": "shot",
            "ntor": 1,
            "omega": 1.0,
            "input_fingerprint": "a" * 64,
            "gap_region": "tae_like",
            "processing_status": "READY_FOR_RULES",
        }

    def test_placeholder_returns_review_with_valid_json(self):
        result = evaluate_mode(self.base)
        row = result.as_output_row(self.base)
        self.assertEqual(row["rule_decision"], "REVIEW")
        self.assertEqual(row["rule_primary_reason"], RULESET_NOT_IMPLEMENTED)
        self.assertEqual(json.loads(row["rule_triggered_rules"]), [RULESET_NOT_IMPLEMENTED])
        self.assertIsNone(json.loads(row["rule_features"])["future_feature_placeholder"])
        self.assertEqual(stable_json({"missing": float("nan")}), '{"missing":null}')

    def test_override_changes_final_but_preserves_rule_decision(self):
        row = evaluate_mode(self.base).as_output_row(self.base)
        override = override_row({key: str(value) for key, value in row.items()})
        final, audit = apply_manual_overrides([row], [override])
        self.assertEqual(final[0]["rule_decision"], "REVIEW")
        self.assertEqual(final[0]["final_decision"], "GOOD")
        self.assertEqual(final[0]["decision_source"], "manual_override")
        self.assertEqual(audit.applied, 1)
        self.assertEqual(audit.decisions_changed, 1)

    def test_all_preliminary_classes_can_be_overridden(self):
        rows = []
        overrides = []
        for index, (rule_decision, manual_decision) in enumerate(
            (("GOOD", "BAD"), ("BAD", "REVIEW"), ("REVIEW", "GOOD")), start=1
        ):
            row = evaluate_mode(self.base).as_output_row(self.base)
            row.update(
                {
                    "path": f"/data/shot/N1/egn01w.{index}",
                    "mode_key": f"shot/N1/egn01w.{index}",
                    "input_fingerprint": str(index) * 64,
                    "rule_decision": rule_decision,
                    "final_decision": rule_decision,
                }
            )
            rows.append(row)
            overrides.append(
                override_row({key: str(value) for key, value in row.items()}, manual_decision)
            )
        final, audit = apply_manual_overrides(rows, overrides)
        self.assertEqual([row["final_decision"] for row in final], ["BAD", "REVIEW", "GOOD"])
        self.assertEqual(audit.applied, 3)

    def test_stale_and_ambiguous_overrides_are_not_applied(self):
        row = evaluate_mode(self.base).as_output_row(self.base)
        stale = override_row({key: str(value) for key, value in row.items()})
        stale["input_fingerprint"] = "b" * 64
        final, audit = apply_manual_overrides([row], [stale])
        self.assertEqual(final[0]["final_decision"], "REVIEW")
        self.assertEqual(final[0]["override_status"], "STALE_FINGERPRINT")
        self.assertEqual(audit.stale, 1)

        valid = override_row({key: str(value) for key, value in row.items()})
        final, audit = apply_manual_overrides([row], [valid, dict(valid)])
        self.assertEqual(final[0]["final_decision"], "REVIEW")
        self.assertEqual(final[0]["override_status"], "AMBIGUOUS_OVERRIDE")
        self.assertEqual(audit.ambiguous, 2)

    def test_summary_counts_only_primary_reason(self):
        row = evaluate_mode(self.base).as_output_row(self.base)
        row["rule_triggered_rules"] = stable_json(
            [RULESET_NOT_IMPLEMENTED, "SECONDARY_AUDIT_CODE"]
        )
        summary = build_summary(
            [row],
            shot="shot",
            selected_paths=frozenset(),
            duplicate_status="SKIPPED_NO_GOOD_MODES",
            override_audit=OverrideAudit(),
            override_sha256="",
            fraction_tae_threshold=0.5,
            fraction_eae_threshold=0.4,
            signed_delta_eae_threshold=-0.1,
            rel_freq_tol=0.02,
        )
        self.assertEqual(
            json.loads(summary["primary_reason_counts_json"]),
            {RULESET_NOT_IMPLEMENTED: 1},
        )

    def test_empty_manual_reason_is_rejected(self):
        with tempfile.TemporaryDirectory() as temporary:
            path = Path(temporary) / "manual_overrides.csv"
            result_row = evaluate_mode(self.base).as_output_row(self.base)
            row = override_row({key: str(value) for key, value in result_row.items()})
            row["manual_reason"] = ""
            write_dict_csv(path, MANUAL_OVERRIDE_FIELDS, [row])
            with self.assertRaisesRegex(ValueError, "manual_reason must be nonempty"):
                load_manual_overrides(path)


class DuplicateHandlingTests(unittest.TestCase):
    def _rows_and_scorer(self, root: Path):
        shot = make_tae_shot(root, names=("egn01w.low", "egn01w.high"))
        paths = [shot / "N1" / "egn01w.low", shot / "N1" / "egn01w.high"]
        rows = [
            {
                "path": str(path),
                "mode_key": portable_mode_key(path),
                "shot": shot.name,
                "ntor": 1,
                "omega": 1.0 + 0.005 * index,
                "final_decision": "GOOD",
                "rule_decision": "REVIEW",
            }
            for index, path in enumerate(paths)
        ]
        review_path = shot / "N1" / "egn01w.review"
        write_mode(review_path, omega=1.007, ntor=1)
        rows.append(
            {
                "path": str(review_path),
                "mode_key": portable_mode_key(review_path),
                "shot": shot.name,
                "ntor": 1,
                "omega": 1.007,
                "final_decision": "REVIEW",
                "rule_decision": "REVIEW",
            }
        )
        calls: list[str] = []

        def scorer(_classifier, path):
            calls.append(path)
            mode, omega, gamma_d, ntor = load_mode_from_nova(path)
            score = 0.9 if path.endswith("high") else 0.1
            return score, mode, omega, gamma_d, ntor

        return rows, paths, calls, scorer

    def test_no_rf_retains_every_good_cluster_member(self):
        with tempfile.TemporaryDirectory() as temporary:
            rows, paths, _calls, _scorer = self._rows_and_scorer(Path(temporary))
            result = deduplicate_final_good(rows, rf_model_path=None, rel_freq_tol=0.02)
        self.assertEqual(result.selected_paths, frozenset(str(path) for path in paths))
        self.assertEqual(result.status, "SKIPPED_NO_RF_CHECKPOINT")
        self.assertEqual(result.cluster_records[0]["status"], "SKIPPED_NO_RF_CHECKPOINT")

    def test_unloadable_rf_retains_every_good_cluster_member(self):
        with tempfile.TemporaryDirectory() as temporary:
            rows, paths, _calls, scorer = self._rows_and_scorer(Path(temporary))

            def failing_loader(_path):
                raise ValueError("synthetic unloadable checkpoint")

            result = deduplicate_final_good(
                rows,
                rf_model_path="bad.joblib",
                rel_freq_tol=0.02,
                rf_loader=failing_loader,
                rf_scorer=scorer,
            )
        self.assertEqual(result.selected_paths, frozenset(str(path) for path in paths))
        self.assertEqual(result.status, "SKIPPED_NO_RF_CHECKPOINT")
        self.assertIn(
            "could not be loaded", result.cluster_records[0]["diagnostic_message"]
        )

    def test_rf_scores_only_final_good_and_keeps_highest_equivalent_mode(self):
        with tempfile.TemporaryDirectory() as temporary:
            rows, paths, calls, scorer = self._rows_and_scorer(Path(temporary))
            decisions_before = [(row["rule_decision"], row["final_decision"]) for row in rows]
            result = deduplicate_final_good(
                rows,
                rf_model_path="fake.joblib",
                rel_freq_tol=0.02,
                rf_loader=lambda _path: object(),
                rf_scorer=scorer,
            )
        self.assertEqual(set(calls), {str(path) for path in paths})
        self.assertEqual(result.selected_paths, frozenset({str(paths[1])}))
        self.assertEqual(result.cluster_records[0]["status"], "PROCESSED_RF")
        self.assertEqual(
            [(row["rule_decision"], row["final_decision"]) for row in rows],
            decisions_before,
        )

    def test_one_rf_scoring_failure_retains_whole_cluster(self):
        with tempfile.TemporaryDirectory() as temporary:
            rows, paths, _calls, scorer = self._rows_and_scorer(Path(temporary))

            def failing_scorer(classifier, path):
                if path.endswith("high"):
                    raise RuntimeError("synthetic score failure")
                return scorer(classifier, path)

            result = deduplicate_final_good(
                rows,
                rf_model_path="fake.joblib",
                rel_freq_tol=0.02,
                rf_loader=lambda _path: object(),
                rf_scorer=failing_scorer,
            )
        self.assertEqual(result.selected_paths, frozenset(str(path) for path in paths))
        self.assertEqual(
            result.cluster_records[0]["status"], "SKIPPED_RF_SCORING_FAILED"
        )


class WorkflowOutputTests(unittest.TestCase):
    def test_complete_output_contract_summary_counts_and_idempotence(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            shot = make_tae_shot(root)
            out_dir = root / "out"
            first = run_shot(shot, out_dir)
            self.assertEqual(set(path.name for path in out_dir.iterdir()), REQUIRED_OUTPUTS)
            for name in REQUIRED_OUTPUTS:
                if name.endswith(".csv"):
                    with (out_dir / name).open(newline="") as handle:
                        self.assertTrue(next(csv.reader(handle)), name)
            self.assertEqual(first.summary["n_preliminary_review"], 1)
            self.assertEqual(first.summary["n_final_review"], 1)
            self.assertEqual(first.summary["n_final_good"], 0)
            self.assertEqual(
                json.loads(first.summary["primary_reason_counts_json"]),
                {RULESET_NOT_IMPLEMENTED: 1},
            )
            _fields, result_rows = read_dict_csv(out_dir / "rule_results.csv")
            self.assertEqual(json.loads(result_rows[0]["rule_triggered_rules"]), [RULESET_NOT_IMPLEMENTED])

            before = {path.name: path.read_bytes() for path in sorted(out_dir.iterdir())}
            run_shot(shot, out_dir)
            after = {path.name: path.read_bytes() for path in sorted(out_dir.iterdir())}
            self.assertEqual(before, after)

    def test_valid_override_hash_and_stale_recheck(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            shot = make_tae_shot(root)
            initial_out = root / "initial"
            run_shot(shot, initial_out)
            _fields, rows = read_dict_csv(initial_out / "rule_results.csv")
            override_path = root / "overrides.csv"
            write_dict_csv(override_path, MANUAL_OVERRIDE_FIELDS, [override_row(rows[0])])

            final_out = root / "final"
            result = run_shot(
                shot, final_out, manual_overrides=override_path, rf_model=None
            )
            final_row = result.final_rows[0]
            self.assertEqual(final_row["rule_decision"], "REVIEW")
            self.assertEqual(final_row["final_decision"], "GOOD")
            self.assertEqual(result.summary["n_final_good"], 1)
            self.assertEqual(result.summary["manual_override_sha256"], sha256_file(override_path))
            self.assertEqual(
                json.loads(result.summary["transition_counts_json"]),
                {"REVIEW->GOOD": 1},
            )

            mode_path = shot / "N1" / "egn01w.one"
            original_mode = mode_path.read_bytes()
            values = np.fromfile(mode_path)
            values[1] *= 1.001
            values.tofile(mode_path)
            stale_mode_out = root / "stale_mode"
            stale_mode_result = run_shot(
                shot, stale_mode_out, manual_overrides=override_path
            )
            self.assertEqual(stale_mode_result.final_rows[0]["final_decision"], "REVIEW")
            self.assertEqual(stale_mode_result.summary["n_stale_overrides"], 1)

            mode_path.write_bytes(original_mode)
            datcon = shot / "N1" / "datcon1"
            datcon.write_text(datcon.read_text() + "\n")
            stale_datcon_out = root / "stale_datcon"
            stale_datcon_result = run_shot(
                shot, stale_datcon_out, manual_overrides=override_path
            )
            self.assertEqual(
                stale_datcon_result.final_rows[0]["final_decision"], "REVIEW"
            )
            self.assertEqual(stale_datcon_result.summary["n_stale_overrides"], 1)

    def test_adjudication_scope_defaults_to_review(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            shot = make_tae_shot(root)
            rows = []
            for index, decision in enumerate(("GOOD", "BAD", "REVIEW", "INVALID")):
                mode = shot / "N1" / f"egn01w.scope{index}"
                row = {field: "" for field in RULE_OUTPUT_FIELDS}
                row.update(
                    {
                        "path": str(mode),
                        "mode_key": portable_mode_key(mode),
                        "input_fingerprint": "a" * 64,
                        "ntor": "1",
                        "omega": "1.0",
                        "rule_decision": decision,
                    }
                )
                rows.append(row)
            source = root / "classifications.csv"
            write_dict_csv(source, RULE_OUTPUT_FIELDS, rows)
            review = adjudication_candidate_rows(
                str(source), mode_dir=shot, data_dir=None, scope="review"
            )
            all_rows = adjudication_candidate_rows(
                str(source), mode_dir=shot, data_dir=None, scope="all"
            )
        self.assertEqual([row["rule_decision"] for row in review], ["REVIEW"])
        self.assertEqual(
            sorted(row["rule_decision"] for row in all_rows), ["BAD", "GOOD", "REVIEW"]
        )


class SkillStructureTests(unittest.TestCase):
    def test_renamed_and_new_skills_have_matching_metadata_and_paths(self):
        old = REPO_ROOT / ".agents/skills/label-tae-like-modes"
        visual = REPO_ROOT / ".agents/skills/visual-tae-rule-development"
        sorter = REPO_ROOT / ".agents/skills/sort-tae-like-modes"
        self.assertFalse(old.exists())
        for skill, name in (
            (visual, "visual-tae-rule-development"),
            (sorter, "sort-tae-like-modes"),
        ):
            skill_text = (skill / "SKILL.md").read_text()
            metadata = (skill / "agents/openai.yaml").read_text()
            self.assertIn(f"name: {name}", skill_text)
            self.assertIn(f"${name}", metadata)
        for script in (
            "prepare_blind_manifest.py",
            "render_blind_diagnostics.py",
            "seal_review.py",
            "compare_reviews.py",
        ):
            self.assertTrue((visual / "scripts" / script).is_file())
        for script in (
            "make_tae_like_list.py",
            "tae_rule_engine.py",
            "sort_shot_rules.py",
            "label_modes_fast.py",
        ):
            self.assertTrue((REPO_ROOT / "scripts" / script).is_file())

    def test_quick_validate_passes_for_both_skills(self):
        configured_root = os.environ.get("CODEX_HOME")
        skill_root = Path(configured_root) if configured_root else Path.home() / ".codex"
        validator = skill_root / "skills/.system/skill-creator/scripts/quick_validate.py"
        if not validator.is_file():
            self.skipTest(f"skill validator is unavailable: {validator}")
        for skill_name in ("visual-tae-rule-development", "sort-tae-like-modes"):
            completed = subprocess.run(
                [
                    sys.executable,
                    str(validator),
                    str(REPO_ROOT / ".agents/skills" / skill_name),
                ],
                check=False,
                capture_output=True,
                text=True,
            )
            self.assertEqual(completed.returncode, 0, completed.stdout + completed.stderr)


if __name__ == "__main__":
    unittest.main()
