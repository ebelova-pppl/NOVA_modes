"""Explicit evidence for joining saved selections to reconstructed continua.

This is an opt-in alternative to an exact historical mode/datcon fingerprint.
It never approves an unlisted mode, scope, or changed reconstruction.
"""
import hashlib
import json
from pathlib import Path


def digest(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda: stream.read(1024*1024), b''):
            h.update(block)
    return h.hexdigest()


class ReconstructionRegistry:
    def __init__(self, path):
        self.path = Path(path).resolve()
        self.sha256 = digest(self.path)
        self.data = json.loads(self.path.read_text())
        if self.data.get('schema') != 'nova-surrogate-reconstruction-v1':
            raise ValueError('Unknown reconstruction evidence schema')
        self.scopes = self.data['scopes']
        self.modes = self.data['modes']
        roots = sorted({str(Path(r['continuum_path']).parent.parent.parent) for r in self.scopes.values()})
        self.continuum_roots = {f'reconstruction_{i+1}':root for i,root in enumerate(roots)}
        for key, record in self.scopes.items():
            parts = Path(key).parts
            if len(parts) != 2 or '..' in parts or Path(key).is_absolute():
                raise ValueError(f'Unsafe reconstruction scope: {key}')
            if record['validation_level'] not in ('historical_centroids', 'historical_crossings', 'historical_settings'):
                raise ValueError(f'Unknown reconstruction validation level: {key}')
            if digest(record['continuum_path']) != record['continuum_sha256']:
                raise ValueError(f'Reconstructed continuum changed: {key}')
        self.verify_sources()

    def verify_sources(self):
        if digest(self.path) != self.sha256:
            raise ValueError('Reconstruction registry changed')
        for path, expected in self.data['source_sha256'].items():
            if digest(path) != expected:
                raise ValueError(f'Reconstruction evidence/input changed: {path}')

    def resolve(self, row, roots, *, zero_selection=False):
        key = row['mode_key']
        scope = str(Path(key).parent)
        evidence = self.modes.get(key)
        record = self.scopes.get(scope)
        if evidence is None or record is None:
            return None
        if zero_selection and not record.get('zero_selection_verified', False):
            return None
        if evidence['selection_input_fingerprint'] != row['input_fingerprint']:
            raise ValueError(f'Reconstruction evidence refers to another selection: {key}')
        continuum = Path(record['continuum_path'])
        if digest(continuum) != record['continuum_sha256']:
            raise ValueError(f'Reconstructed continuum changed: {scope}')
        for label, root in roots:
            mode = root/key
            if mode.is_file() and digest(mode) == evidence['mode_sha256']:
                return 'reconstructed:'+label, mode, continuum
        return None

    def evidence_for(self, scope):
        return dict(self.scopes[scope], registry_sha256=self.sha256,
                    pairing='historical_reconstruction; original combined fingerprint is retained separately',
                    convergence='not certified')

    def continuum_label(self, path):
        root = str(Path(path).parent.parent.parent)
        return next(label for label, value in self.continuum_roots.items() if value==root)


class ConfirmedEmptyRegistry:
    """User-confirmed NOVA upstream zeros, scoped to specific case/n inputs.

    Absence of classifier rows is expected when NOVA's own numerical-mode
    filter leaves no egn files. This evidence is distinct from classifier K=0.
    """
    STATUS = 'available_nova_filtered_zero'
    CHANGED_FIELDS = ('status', 'status_detail', 'continuum_status',
                      'mode_targets_valid', 'K', 'has_good_tae')

    def __init__(self, path):
        self.path = Path(path).resolve()
        self.sha256 = digest(self.path)
        self.data = json.loads(self.path.read_text())
        if self.data.get('schema') != 'nova-confirmed-empty-v1':
            raise ValueError('Unknown confirmed-empty evidence schema')
        self.scopes = self.data['scopes']
        if not self.data.get('confirmation', {}).get('statement') or not self.scopes:
            raise ValueError('Confirmed-empty evidence requires a statement and named scopes')
        for scope in self.scopes:
            parts = Path(scope).parts
            if (len(parts) != 2 or Path(scope).is_absolute() or '..' in parts
                    or parts[1] not in {f'N{n}' for n in range(1, 11)}):
                raise ValueError(f'Invalid confirmed-empty scope: {scope}')

    def verify_scope(self, scope, data_root):
        directory = Path(data_root)/scope
        # Missing directories and broken egn links must not become zero labels.
        if not directory.is_dir() or any(directory.glob('egn*')):
            raise ValueError(f'Confirmed-empty scope is missing or contains mode entries: {scope}')
        n = int(Path(scope).name[1:])
        if digest(directory/f'datcon{n}') != self.scopes[scope]['continuum_sha256']:
            raise ValueError(f'Confirmed-empty continuum changed: {scope}')

    def verify_sources(self, data_root):
        if digest(self.path) != self.sha256:
            raise ValueError('Confirmed-empty registry changed')
        for scope in self.scopes:
            self.verify_scope(scope, data_root)

    def apply(self, group, data_root):
        """Promote only an available, unheld missing-selection record to K=0."""
        shot = group.parent.name.rsplit('/', 1)[-1]
        scope = f'{shot}/N{int(group["n_tor"][()])}'
        if scope not in self.scopes or group['status'].asstr()[()] != 'missing_selection':
            return False
        if not group['continuum_available'][()]:
            return False
        self.verify_scope(scope, data_root)
        if group['continuum_sha256'].asstr()[()] != self.scopes[scope]['continuum_sha256']:
            raise ValueError(f'Exported continuum differs from confirmed-empty evidence: {scope}')
        if (group['K'][()] != -1 or group['catalog_K'][()] != -1
                or any(len(ds) for ds in group['modes'].values())):
            raise ValueError(f'Confirmed-empty scope has conflicting selection data: {scope}')
        import h5py
        evidence = dict(self.scopes[scope], confirmation=self.data['confirmation'],
                        registry_sha256=self.sha256, scope=scope,
                        mode_file_count=0, classifier_run=False,
                        basis='NOVA internal numerical-mode filter; user-confirmed zero survivors')
        group['status'][()] = self.STATUS
        group['status_detail'][()] = evidence['basis']
        group['continuum_status'][()] = 'available_upstream_zero'
        group['mode_targets_valid'][()] = True
        group['K'][()] = 0
        group['has_good_tae'][()] = 0
        group.create_dataset('zero_K_provenance', data='nova_internal_filter', dtype=h5py.string_dtype())
        group.create_dataset('zero_K_evidence_json', data=json.dumps(evidence, sort_keys=True),
                             dtype=h5py.string_dtype())
        # catalog_K stays -1: there was no downstream classifier input/summary.
        return True

    def write_provenance(self, hdf5):
        import h5py
        provenance = hdf5['provenance']
        provenance.create_dataset('confirmed_empty_registry_json', data=json.dumps(self.data, sort_keys=True),
                                  dtype=h5py.string_dtype())
        provenance.create_dataset('confirmed_empty_registry_sha256', data=self.sha256,
                                  dtype=h5py.string_dtype())
        hdf5.attrs['upstream_zero_convention'] = (
            'Named empty mode directories confirmed by user as zero NOVA-filter survivors; '
            'K=0 without a downstream classifier summary; existing holds remain active')
