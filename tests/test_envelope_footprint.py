import unittest
import numpy as np

from envelope_footprint import measure_envelope_footprint, footprint_exception_qualifies
from tae_rule_engine import (
    evaluate_mode, InteriorUnresolvedEnvelopeConfig, ContinuumCrossingConfig,
    BAD_AXIS_SPIKE, BAD_INTERIOR_UNRESOLVED_ENVELOPE,
)
import test_rule_sorting as fixtures


class EnvelopeFootprintTests(unittest.TestCase):
    def test_strict_limits_and_unresolved_measurements(self):
        for f, q, expected in ((.49, .049, True), (.5, .049, False), (.49, .05, False)):
            self.assertEqual(footprint_exception_qualifies(
                dict(status="MEASURED", spikes_fraction=f, local_hf_fraction=q), .5, .05), expected)
        coarse = np.zeros((2, 11)); coarse[0, 3] = 1
        result = measure_envelope_footprint(coarse)
        self.assertEqual(result["status"], "WINDOW_UNRESOLVED")
        self.assertFalse(footprint_exception_qualifies(result, .5, .05))

    def test_disjoint_components_not_double_counted_by_overlapping_windows(self):
        mode = np.zeros((2, 201)); mode[0, 20] = 1; mode[1, 23] = np.sqrt(.7)
        f = measure_envelope_footprint(mode)
        self.assertEqual(f["n_narrow_halfmax_components"], 2)
        # Both regions use half the GLOBAL maximum, so the lower peak's
        # integration bounds are tighter than its own half-height bounds.
        self.assertAlmostEqual(f["spikes_fraction"], 9/14)
        self.assertGreater(f["windows"][0]["last"], f["windows"][1]["first"])

    def test_integrated_exception_and_other_gate_precedence(self):
        fixture = fixtures.RuleAndOverrideTests(); fixture.setUp()
        r = np.linspace(0, 1, 201)
        mode = np.zeros((3, 201))
        mode[0] = .2 * np.exp(-((r - .6)/.3)**2)
        mode[1] = np.exp(-((r - .25)/.0075)**2)
        options = dict(low2=np.full(201, .25), high2=np.full(201, 2.25),
                       continuum_crossing_config=ContinuumCrossingConfig(w_cross_threshold=None))
        active = evaluate_mode(fixture.base, mode=mode, **options)
        legacy = evaluate_mode(fixture.base, mode=mode, **options,
            interior_unresolved_envelope_config=InteriorUnresolvedEnvelopeConfig(footprint_spikes_fraction_max=None))
        self.assertEqual(legacy.primary_reason, BAD_INTERIOR_UNRESOLVED_ENVELOPE)
        self.assertEqual(active.decision, "REVIEW")
        e = active.features["resolution_features"]["interior_unresolved_envelope"]
        self.assertTrue(e["footprint_exception"]["applied"])
        self.assertEqual(active.features["severity_features"]["gates"][BAD_INTERIOR_UNRESOLVED_ENVELOPE]["severity"], 0)
        mode[2, 1] = .4
        result = evaluate_mode(fixture.base, mode=mode, **options)
        self.assertEqual(result.primary_reason, BAD_AXIS_SPIKE)


if __name__ == "__main__":
    unittest.main()
