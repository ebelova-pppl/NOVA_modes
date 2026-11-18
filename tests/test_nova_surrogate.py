"""Scientific-coordinate and provenance regressions for the HDF5 exporter."""
import csv
import json
from pathlib import Path
import sys
import tempfile
import unittest

import h5py
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]/'scripts'))
from export_nova_surrogate import mode_moments, resolve_pair, write_scope, zero_selection_anchors
from nova_equilibrium import density_spline, q_axis_and_minimum, read_transp, processed_density
from tae_rule_io import input_fingerprint
from nova_surrogate_provenance import ConfirmedEmptyRegistry, ReconstructionRegistry, digest


class CoordinateTests(unittest.TestCase):
    def test_psi_moments_are_recomputed_from_amplitudes(self):
        # Equal weight at r=0 and r=1: both coordinate widths are 0.5.
        m = np.array([[1., 0., 1.], [0., 0., 0.]])
        result = mode_moments(m)
        for key in ('r_loc', 'r_width', 'psi_N_loc', 'psi_N_width'):
            self.assertAlmostEqual(result[key], .5)
        self.assertNotAlmostEqual(result['psi_N_width'], result['r_width']**2)
        self.assertEqual(result, mode_moments(-m))

    def test_cubic_density_and_endpoint_derivatives(self):
        x = np.array([0., .03, .2, .55, .8, 1.])
        profile = lambda x: 1 + .2*x - .4*x*x + .1*x**3
        sample = np.linspace(0, 1, 101)
        np.testing.assert_allclose(density_spline(x, profile(x))(sample), profile(sample), atol=1e-13)

    def test_axis_and_reversed_shear_minimum(self):
        psi = np.array([2.5e-7, .000025, .25, 1.])
        q0, qmin, location = q_axis_and_minimum(psi, 2+psi)
        self.assertAlmostEqual(q0, 2.)
        self.assertEqual(q0, qmin)
        self.assertEqual(location, 0.)
        _, qmin, location = q_axis_and_minimum(psi, np.array([2., 1.9999, 1.5, 4.]))
        self.assertEqual((qmin, location), (1.5, .25))

    def test_missing_species_is_not_silently_zero_filled(self):
        with self.assertRaisesRegex(ValueError, 'Missing/wrong length'):
            processed_density({'X': np.arange(4.)})


class ScopeTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.root = Path(self.tmp.name)
        self.live = self.root/'live'
        self.snapshot = self.root/'snapshot'
        self.key = 'case/N1/egn01w.test'
        self.path = self.live/self.key
        self.path.parent.mkdir(parents=True)
        # Stored array order: field, harmonic, radius; frequency is already omega.
        fields = np.zeros((3, 2, 5))
        fields[0, 0, [1, 3]] = [1., -.5]
        np.r_[2.5, fields.ravel(), 5., 0., 1.].astype('f8').tofile(self.path)
        self.datcon = self.path.parent/'datcon1'
        self.datcon.write_text('1 5\n'+'4 9\n'*5)
        self.row = dict(mode_key=self.key, shot='case', ntor='1',
                        input_fingerprint=input_fingerprint(self.path, self.datcon))
        self.catalog = {self.key: dict(omega='2.5', family='TAE', review_status='pending',
                                      rule_config='test', manual_override='no')}

    def tearDown(self):
        self.tmp.cleanup()

    def scope(self, *, rows=None, summaries=None, hold='', anchors=None, reconstruction=None, confirmed_empty=None):
        h = h5py.File(self.root/'test.h5', 'w')
        g = h.create_group('equilibria/case/n_01')
        write_scope(g, ntor=1, nr=5, rows=[self.row] if rows is None else rows,
                    catalog=self.catalog, summaries={1:{'n_final_good':'1'}} if summaries is None else summaries,
                    roots=[('live', self.live), ('snapshot', self.snapshot)], hold=hold,
                    equilibrium_ok=True, anchors=anchors or {}, reconstruction=reconstruction,
                    confirmed_empty=confirmed_empty)
        return h, g

    def reconstruction(self, zero=False):
        continuum = self.root/'restored/cases/case/N1/datcon1'
        continuum.parent.mkdir(parents=True, exist_ok=True)
        continuum.write_text('1 5\n'+'16 25\n'*5)
        self.datcon.write_text('1 5\n'+'25 36\n'*5)
        manifest = self.root/'reconstruction.json'
        manifest.write_text(json.dumps(dict(schema='nova-surrogate-reconstruction-v1',source_sha256={},
            scopes={'case/N1':dict(continuum_path=str(continuum),continuum_sha256=digest(continuum),
                validation_level='historical_settings',zero_selection_verified=zero)},
            modes={self.key:dict(mode_sha256=digest(self.path),selection_input_fingerprint=self.row['input_fingerprint'])})))
        return ReconstructionRegistry(manifest), continuum

    def test_frequency_conversion_and_array_order(self):
        h, g = self.scope()
        with h:
            np.testing.assert_equal(g['omega_minus'][:], 2.)
            np.testing.assert_equal(g['omega_plus'][:], 3.)
            np.testing.assert_equal(g['psi_N'][:], g['r'][:]**2)
            self.assertEqual(g['modes/omega'][0], 2.5)
            self.assertAlmostEqual(g['modes/r_loc'][0], .35)
            self.assertEqual(g['K'][()], 1)
            self.assertTrue(g['mode_targets_valid'][()])

    def test_snapshot_recovers_exact_pair_after_live_continuum_changes(self):
        target = self.snapshot/self.key
        target.parent.mkdir(parents=True)
        target.write_bytes(self.path.read_bytes())
        (target.parent/'datcon1').write_bytes(self.datcon.read_bytes())
        self.datcon.write_text('1 5\n'+'16 25\n'*5)
        self.assertEqual(resolve_pair(self.row, [('live', self.live), ('snapshot', self.snapshot)])[0], 'snapshot')
        h, g = self.scope()
        with h:
            self.assertEqual(g['continuum_source_root'].asstr()[()], 'snapshot')
            np.testing.assert_equal(g['omega_minus'][:], 2.)

    def test_stale_pair_is_not_exported_as_no_tae(self):
        self.datcon.write_text('1 5\n'+'16 25\n'*5)
        h, g = self.scope()
        with h:
            self.assertEqual(g['K'][()], -1)
            self.assertEqual(g['has_good_tae'][()], -1)
            self.assertEqual(len(g['modes/omega']), 0)
            self.assertEqual(g['continuum_status'].asstr()[()], 'available_unpaired')
            self.assertTrue(g['continuum_available'][()])

    def test_zero_K_requires_summary_and_matching_anchor(self):
        h, g = self.scope(rows=[], summaries={1:{'n_final_good':'0'}}, anchors={1:[self.row]})
        with h:
            self.assertEqual(g['K'][()], 0)
            self.assertEqual(g['has_good_tae'][()], 0)
        h, g = self.scope(rows=[], summaries={})
        with h:
            self.assertEqual(g['K'][()], -1)
            self.assertEqual(g['status'].asstr()[()], 'missing_selection')

    def test_hold_masks_targets_even_if_pair_matches(self):
        h, g = self.scope(hold='Known invalid input')
        with h:
            self.assertEqual(g['K'][()], -1)
            self.assertFalse(g['continuum_available'][()])
            self.assertTrue(np.isnan(g['omega_minus'][:]).all())

    def test_invalid_diagnostic_with_missing_n_is_not_a_zero_K_anchor(self):
        path = self.root/'diagnostics.csv'
        fields = ['mode_key', 'shot', 'ntor', 'input_fingerprint', 'processing_status']
        with path.open('w', newline='') as stream:
            writer = csv.DictWriter(stream, fieldnames=fields)
            writer.writeheader()
            writer.writerow({**self.row, 'ntor': '', 'processing_status': 'INVALID'})
            writer.writerow({**self.row, 'processing_status': 'RULE_EVALUATED'})
        self.assertEqual(zero_selection_anchors(path, {1}), {1:[self.row]})

    def test_reconstruction_keeps_old_and_current_fingerprints_distinct(self):
        registry, continuum = self.reconstruction()
        h,g = self.scope(reconstruction=registry)
        with h:
            self.assertEqual(g['status'].asstr()[()], 'available_reconstructed')
            self.assertEqual(g['K'][()], 1)
            np.testing.assert_equal(g['omega_minus'][:], 4.)
            self.assertEqual(g['modes/input_fingerprint'].asstr()[0], input_fingerprint(self.path,continuum))
            self.assertEqual(g['modes/selection_input_fingerprint'].asstr()[0], self.row['input_fingerprint'])
            self.assertNotEqual(g['modes/input_fingerprint'].asstr()[0], self.row['input_fingerprint'])
            self.assertEqual(g['modes/pair_provenance'].asstr()[0], 'reconstructed_continuum')

    def test_changed_mode_cannot_use_reconstructed_pair(self):
        registry, _ = self.reconstruction()
        values=np.fromfile(self.path);values[1]+=0.1;values.tofile(self.path)
        h,g = self.scope(reconstruction=registry)
        with h:
            self.assertEqual(g['K'][()], -1)
            self.assertFalse(g['mode_targets_valid'][()])

    def test_changed_reconstructed_continuum_is_rejected(self):
        registry, continuum = self.reconstruction()
        continuum.write_text('1 5\n'+'36 49\n'*5)
        with self.assertRaisesRegex(ValueError,'Reconstructed continuum changed'):
            resolve_pair(self.row,[('live',self.live)],registry)

    def test_reconstructed_zero_requires_its_own_audit(self):
        registry,_ = self.reconstruction(zero=False)
        h,g = self.scope(reconstruction=registry, rows=[], summaries={1:{'n_final_good':'0'}}, anchors={1:[self.row]})
        with h:self.assertEqual(g['K'][()], -1)
        registry,_ = self.reconstruction(zero=True)
        h,g = self.scope(reconstruction=registry, rows=[], summaries={1:{'n_final_good':'0'}}, anchors={1:[self.row]})
        with h:
            self.assertEqual(g['K'][()], 0)
            self.assertTrue(g['mode_targets_valid'][()])

    def test_hold_remains_masked_with_reconstruction_evidence(self):
        registry,_ = self.reconstruction()
        h,g = self.scope(reconstruction=registry, hold='Existing input hold')
        with h:
            self.assertEqual(g['K'][()], -1)
            self.assertFalse(g['continuum_available'][()])

    def empty_registry(self):
        path = self.root/'empty.json'
        path.write_text(json.dumps(dict(schema='nova-confirmed-empty-v1',
            confirmation={'statement':'User confirms NOVA filtering left zero mode files'},
            scopes={'case/N1':{'continuum_sha256':digest(self.datcon)}})))
        return ConfirmedEmptyRegistry(path)

    def test_confirmed_nova_zero_preserves_continuum_without_classifier_summary(self):
        registry = self.empty_registry()
        self.path.unlink()
        h,g = self.scope(rows=[], summaries={}, confirmed_empty=registry)
        with h:
            self.assertEqual(g['K'][()], 0)
            self.assertEqual(g['has_good_tae'][()], 0)
            self.assertEqual(g['catalog_K'][()], -1)
            self.assertTrue(g['mode_targets_valid'][()])
            self.assertEqual(g['status'].asstr()[()], registry.STATUS)
            self.assertEqual(g['zero_K_provenance'].asstr()[()], 'nova_internal_filter')
            self.assertEqual(len(g['modes/omega']), 0)
            np.testing.assert_equal(g['omega_minus'][:], 2.)
            np.testing.assert_equal(g['omega_plus'][:], 3.)

    def test_new_mode_file_invalidates_confirmed_empty_evidence(self):
        registry = self.empty_registry()
        h,g = self.scope(rows=[], summaries={})
        with h:
            with self.assertRaisesRegex(ValueError, 'contains mode entries'):
                registry.apply(g,self.live)
            self.assertEqual(g['K'][()], -1)

    def test_missing_directory_is_not_a_confirmed_nova_zero(self):
        registry = self.empty_registry()
        h,g = self.scope(rows=[], summaries={})
        self.path.parent.rename(self.path.parent.with_name('moved'))
        with h:
            with self.assertRaisesRegex(ValueError, 'missing or contains'):
                registry.apply(g,self.live)
            self.assertEqual(g['K'][()], -1)

    def test_confirmed_nova_zero_cannot_release_a_hold(self):
        registry = self.empty_registry()
        self.path.unlink()
        h,g = self.scope(rows=[], summaries={}, hold='Existing input hold', confirmed_empty=registry)
        with h:
            self.assertEqual(g['K'][()], -1)
            self.assertEqual(g['status'].asstr()[()], 'held')

    def test_confirmed_nova_zero_requires_available_continuum(self):
        registry = self.empty_registry()
        self.path.unlink()
        self.datcon.unlink()
        h,g = self.scope(rows=[], summaries={}, confirmed_empty=registry)
        with h:
            self.assertEqual(g['K'][()], -1)
            self.assertFalse(g['mode_targets_valid'][()])

    def test_confirmed_nova_zero_rejects_changed_continuum(self):
        registry = self.empty_registry()
        self.path.unlink()
        self.datcon.write_text('1 5\n'+'16 25\n'*5)
        h,g = self.scope(rows=[], summaries={})
        with h:
            with self.assertRaisesRegex(ValueError, 'continuum changed'):
                registry.apply(g,self.live)
            self.assertEqual(g['K'][()], -1)


if __name__ == '__main__':
    unittest.main()
