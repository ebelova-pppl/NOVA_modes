"""Severity margins and deterministic representative selection without RF."""
import contextlib
import copy
import csv
import io
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import numpy as np

from rule_severity import component, combined, gate_record, GATE_NAMES
from tae_rule_engine import evaluate_mode, ContinuumCrossingConfig, AxisEnergyConcentrationConfig
from tae_rule_config import load_rule_run_configuration, PRODUCTION_RULE_CONFIG_NAME
from sort_shot_rules import run_configured_shot, deduplicate_final_good
from test_rule_sorting import write_mode, write_datcon


def evaluate(mode, **kwargs):
    nr = mode.shape[1]
    return evaluate_mode(dict(path='shot/N1/egn01w.test', mode_key='shot/N1/egn01w.test',
                              shot='shot', ntor=1, omega=1., gamma_d=0,
                              input_fingerprint='a'*64, gap_region='tae_like'),
                         mode=mode, low2=np.full(nr, .25), high2=np.full(nr, 2.25),
                         continuum_crossing_config=ContinuumCrossingConfig(w_cross_threshold=None),
                         **kwargs)


def smooth_mode():
    r = np.linspace(0, 1, 201)
    mode = np.zeros((4, 201))
    mode[1] = np.exp(-((r-.65)/.08)**2)
    mode[2] = .5*mode[1]
    return mode


class SeverityTests(unittest.TestCase):
    def test_boolean_components_unclipped_and_unknown(self):
        parts = dict(roughness=component(.6, .2), length=component(.0288, .04))
        self.assertAlmostEqual(combined(parts), .72)
        self.assertAlmostEqual(combined(parts, 'OR'), 3)
        record = gate_record(True, False, [(parts, {'region': 1})])
        self.assertEqual(record['witness'], {'region': 1})
        self.assertFalse(record['fired'])
        record = gate_record(True, True, [({'amplitude': component(.9, .3)}, None)])
        self.assertAlmostEqual(record['severity'], 3)
        self.assertEqual(gate_record(True, False)['severity'], 0)
        self.assertIsNone(gate_record(False, False)['severity'])
        self.assertIsNone(gate_record(True, False, unavailable='UNSUPPORTED_RESOLUTION')['severity'])
        for value, threshold in ((None, 1), (1, 0), (0, 0)):
            self.assertIsNone(component(value, threshold)['ratio'])
        self.assertEqual(component(0, 1, small_is_bad=True)['ratio'], 1e12)

    def test_all_gates_output_and_inclusive_axis_boundary(self):
        mode = smooth_mode(); mode[0, 2] = .2
        result = evaluate(mode)
        report = result.features['severity_features']
        self.assertEqual(set(report['gates']), set(GATE_NAMES))
        gate = report['gates']['BAD_AXIS_SPIKE']
        self.assertEqual(gate['severity'], 1)
        self.assertTrue(gate['fired'])
        self.assertEqual(result.primary_reason, 'BAD_AXIS_SPIKE')
        row = result.as_output_row()
        self.assertEqual(row['gate_severity_BAD_AXIS_SPIKE'], 1)
        self.assertEqual(row['rule_margin'], 1-row['overall_rule_severity'])
        self.assertEqual(len(row['severity_config_sha256']), 64)
        mode[0, 2] = .1
        result = evaluate(mode)
        self.assertAlmostEqual(result.features['severity_features']['gates']['BAD_AXIS_SPIKE']['severity'], .5)
        self.assertEqual(result.decision, 'REVIEW')

    def test_width_and_amplitude_belong_to_same_peak(self):
        mode = smooth_mode()
        # A broad high-amplitude axis lobe and a narrow weak one cannot supply
        # separate maxima to fabricate a rejecting severity.
        mode[0] = .9*np.exp(-(np.linspace(0, 1, 201)/.12)**2)
        mode[3, 2] = .1
        result = evaluate(mode)
        gate = result.features['severity_features']['gates']['BAD_AXIS_SPIKE']
        self.assertFalse(gate['fired'])
        self.assertLess(gate['severity'], 1)
        self.assertGreater(gate['severity'], .4)

    def test_weak_packet_has_continuous_step_margin(self):
        mode = smooth_mode(); mode[0, 40:45] = [0, .15, -.15, .15, 0]
        result = evaluate(mode)
        gate = result.features['severity_features']['gates']['BAD_GRID_SCALE_PACKET']
        self.assertFalse(gate['fired'])
        self.assertAlmostEqual(gate['severity'], .5)
        self.assertEqual(gate['witness']['required_turns'], 3)
        self.assertAlmostEqual(gate['components']['turn_step']['ratio'], .75)

    def test_strict_axis_energy_boundary_and_disabled_gate(self):
        from test_axis_energy_concentration import long_axis_shoulder
        mode = long_axis_shoulder()
        f = evaluate(mode).features['boundary_features']['axis_energy_concentration']
        result = evaluate(mode, axis_energy_concentration_config=AxisEnergyConcentrationConfig(
            amplitude_min=f['axis_amplitude'], energy_fraction_min=f['inner_energy_fraction']))
        gate = result.features['severity_features']['gates']['BAD_AXIS_ENERGY_CONCENTRATION']
        self.assertEqual(gate['severity'], 1)
        self.assertFalse(gate['fired'])
        report = evaluate(mode, axis_energy_concentration_config=AxisEnergyConcentrationConfig(
            amplitude_min=None)).features['severity_features']
        self.assertIsNone(report['gates']['BAD_AXIS_ENERGY_CONCENTRATION']['severity'])
        self.assertNotEqual(report['nearest_gate'], 'BAD_AXIS_ENERGY_CONCENTRATION')

    def test_current_preset_no_model_load_legacy_and_ties(self):
        for version in range(5, 11):
            self.assertEqual(load_rule_run_configuration(f'tae_rules_production_v{version}').run_kwargs['duplicate_rank_method'], 'rf_p_good')
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp); shot=root/'shot'; mode=smooth_mode()
            # Structure-identical modes with different small axis defects.
            for name, amplitude in [('a', .1), ('b', .05)]:
                a=mode.copy(); a[0,2]=amplitude
                write_mode(shot/f'N1/egn01w.{name}', omega=1., ntor=1, nr=201, mode=a)
            write_datcon(shot/'N1/datcon1', nr=201, upper_frequency=1.5)
            with patch('joblib.load', side_effect=AssertionError('RF must not be loaded')):
                result=run_configured_shot(shot, root/'out', rule_config=PRODUCTION_RULE_CONFIG_NAME,
                                           rf_model=root/'absent.joblib', rule_survivor_policy='accept-as-good-v1')
            self.assertEqual(result.summary['duplicate_processing_status'], 'COMPLETED_RULE_SEVERITY')
            selected=[r for r in result.final_rows if r['selected_final']]
            self.assertEqual(len(selected), 1)
            self.assertTrue(selected[0]['path'].endswith('.b'))
            self.assertEqual(selected[0]['duplicate_rank_source'], 'rule_severity')
            with (root/'out/good_tae_final.csv').open() as f:
                saved=next(csv.DictReader(f))
            self.assertEqual(saved['rule_configuration_sha256'], result.summary['rule_configuration_sha256'])
            self.assertIn('overall_rule_severity', saved)
            rows=copy.deepcopy(result.final_rows)
            for r in rows:r['overall_rule_severity']=.4
            dedup=deduplicate_final_good(list(reversed(rows)), rf_model_path=None, rel_freq_tol=.02, rank_method='rule_severity')
            self.assertEqual(dedup.selected_paths, frozenset([str(shot/'N1/egn01w.a')]))
            rows[0]['severity_complete']=False
            dedup=deduplicate_final_good(rows, rf_model_path=None, rel_freq_tol=.02, rank_method='rule_severity')
            self.assertEqual(len(dedup.selected_paths), 2)
            self.assertEqual(dedup.status, 'COMPLETED_WITH_SEVERITY_FALLBACK')


if __name__ == '__main__':
    unittest.main()
