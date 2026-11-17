"""Mapped NOVA equilibrium scalars and the confirmed TRANSP iden=4 density.

rho is physical ion mass density in kg/m^3 (without mu0). Arrays are on
normalized poloidal flux. No NOVA-C density or diagnostic B0 placeholder is used.
"""
from pathlib import Path
import re

import numpy as np
from scipy.constants import proton_mass
from scipy.interpolate import CubicSpline
from scipy.io import netcdf_file

MU0 = 4 * np.pi * 1e-7  # Constant used by this NOVA source.
HEADER = re.compile(r'^\s*(\d+)\s+([A-Z][A-Z0-9_]*)\s+.*:')
SPECIES = ('NH', 'ND', 'NT', 'NIMP', 'NFI', 'BDENS_D', 'BDENS_T', 'NHE3')
SMOOTHED = ('NH', 'ND', 'NT', 'NIMP', 'NFI', 'BDENS_D', 'BDENS_T')
REQUIRED = ('X', 'XB', 'PLFLX', 'NE', 'AIMPJ', 'XZIMPJ') + SPECIES


def read_transp(path):
    lines = path.read_text().splitlines()
    blocks, headers = {}, {}
    for i, line in enumerate(lines):
        match = HEADER.match(line)
        if not match:
            continue
        n, name = int(match[1]), match[2]
        if name in blocks:
            raise ValueError(f'Duplicate TRANSP block {name}')
        headers[name] = line.strip()
        if n <= 1:
            vals = [float(line.split()[2].replace('D', 'E'))]
        else:
            vals = []
            for data_line in lines[i + 1:]:
                if HEADER.match(data_line):
                    break
                vals.extend(float(v.replace('D', 'E')) for v in data_line.split())
                if len(vals) >= n:
                    break
            if len(vals) != n:
                raise ValueError(f'{name}: declared {n} values, found {len(vals)}')
        blocks[name] = np.asarray(vals)
        if not np.all(np.isfinite(blocks[name])):
            raise ValueError(f'{name}: nonfinite values')
    return blocks, headers


def smooth(y):
    z = y.copy()
    z[1:-1] = (y[:-2] + y[1:-1] + y[2:]) / 3
    z[0], z[-1] = (y[0] + y[1]) / 2, (y[-2] + y[-1]) / 2
    return z


def extrap1(y, x, edge):
    z = y.copy()
    z[0] -= (y[1] - y[0]) / (x[1] - x[0]) * x[0]
    z[-1] += (y[-1] - y[-2]) / (x[-1] - x[-2]) * (edge - x[-1])
    return z


def boundary_to_center(xb, y, xc):
    z = np.empty_like(y)
    for i, x in enumerate(xc):
        start = min(max(0, i - 2), len(xb) - 4)
        nodes = xb[start:start + 4]
        value = 0.
        for j in range(4):
            others = np.arange(4) != j
            value += y[start + j] * np.prod((x - nodes[others]) / (nodes[j] - nodes[others]))
        z[i] = value
    return z


def processed_density(blocks):
    n = len(blocks['X'])
    for name in REQUIRED:
        if name not in blocks or len(blocks[name]) != n:
            raise ValueError(f'Missing/wrong length density block {name}')
    x, xb = blocks['X'], blocks['XB']
    if n < 4 or not np.all(np.diff(x) > 0) or not np.all(np.diff(xb) > 0):
        raise ValueError('Invalid TRANSP radial grids')
    d = {k: smooth(blocks[k]) if k in SMOOTHED else blocks[k] for k in SPECIES}
    mass = (d['NH'] + 2*d['ND'] + 3*d['NT'] + 2*d['BDENS_D']
            + 3*d['BDENS_T'] + 4*d['NFI'] + 3*d['NHE3']
            + blocks['AIMPJ']*d['NIMP'])
    mass = extrap1(mass, x, xb[-1])
    flux = extrap1(boundary_to_center(xb, blocks['PLFLX'], x), x, xb[-1])
    psi = (flux - flux[0]) / (flux[-1] - flux[0])
    if not np.all(np.isfinite(psi)) or not np.all(np.diff(psi) > 0):
        raise ValueError('Nonmonotone processed poloidal flux')
    return psi, mass * proton_mass * 1e6  # cm^-3 -> m^-3; A=m/m_proton.


def geometry(r, z, zmid):
    # Piecewise-linear contour intersections; handles vertices on the plane.
    crossings = []
    for i in range(len(r) - 1):
        za, zb = z[i] - zmid, z[i + 1] - zmid
        if abs(za) < 1e-10:
            crossings.append(r[i])
        if za * zb < 0:
            crossings.append(r[i] - za * (r[i + 1] - r[i]) / (zb - za))
    crossings = np.unique(np.round(crossings, 12))
    if len(crossings) != 2:
        raise ValueError(f'Expected two midplane crossings, found {crossings}')
    rin, rout = crossings
    a, rv = (rout - rin) / 2, (rout + rin) / 2
    up, down = np.argmax(z), np.argmin(z)
    du, dl = (rv - r[up]) / a, (rv - r[down]) / a
    return dict(a_m=a, R_v_m=rv, R_in_midplane_m=rin, R_out_midplane_m=rout,
                kappa=(z[up] - z[down]) / (2*a), delta=(du + dl) / 2,
                delta_upper=du, delta_lower=dl, Z_midplane_m=zmid)


PARAMETERS = {
    'a': ('m', 'Half the LCFS width at Zmag'),
    'R_v': ('m', 'Midpoint of LCFS intersections with Z=Zmag'),
    'delta': ('1', 'Mean upper/lower mapped LCFS triangularity relative to R_v'),
    'kappa': ('1', 'Mapped LCFS height divided by 2a'),
    'B_v': ('T', 'Magnitude of TRANSP BZXR/(100 R_v)'),
    'q_0': ('1', 'Mapped qa extrapolated linearly in psi_N to zero'),
    'q_lcfs': ('1', 'Last mapped qa sample'),
    'q_min': ('1', 'Minimum of axis estimate and mapped qa samples'),
    'p_axis': ('Pa', 'First mapped pa sample times B_v^2/mu0; near-axis proxy'),
    'rho_axis': ('kg m^-3', 'NOVA iden=4 ion mass sum extrapolated to axis'),
    'delta_axis': ('1', '(Xmag-R_v)/a'),
    'psi_N_qmin': ('1', 'Flux at q_min; exactly zero for an axis minimum'),
    'rho_half_over_axis': ('1', 'NOVA spline rho(psi_N=0.5)/rho_axis'),
}


def density_spline(psi, rho):
    """Cubic interpolant with the endpoint derivatives used by NOVA depose.

    depose estimates each derivative from four endpoint samples, then solves
    the clamped cubic spline. Its special large-axis-outlier branch is rejected
    here rather than silently changing the axis normalization.
    """
    psi, rho = np.asarray(psi, dtype=float), np.asarray(rho, dtype=float)
    if (psi.ndim != 1 or rho.shape != psi.shape or len(psi) < 4
            or not np.all(np.isfinite(rho)) or not np.all(rho > 0)
            or not np.all(np.diff(psi) > 0) or psi[0] != 0 or psi[-1] != 1):
        raise ValueError('Density spline requires positive samples on increasing psi_N in [0,1]')
    if abs(rho[0]) > abs(10 * rho[1]):
        raise ValueError('NOVA depose axis-outlier branch requires separate verification')
    # Newton divided differences, identical endpoint conditions to depose.
    x, y = psi[:4], rho[:4]
    d1 = np.diff(y) / np.diff(x)
    d2 = np.diff(d1) / (x[2:] - x[:-2])
    d3 = (d2[1] - d2[0]) / (x[3] - x[0])
    left = d1[0] - d2[0] * x[1] + d3 * x[1] * x[2]
    x, y = psi[-4:], rho[-4:]
    d1 = np.diff(y) / np.diff(x)
    d2 = np.diff(d1) / (x[2:] - x[:-2])
    d3 = (d2[1] - d2[0]) / (x[3] - x[0])
    right = x[3] * (d1[2] + (x[3] - x[2]) * (d2[1] + d3 * (x[3] - x[1])))
    return CubicSpline(psi, rho, bc_type=((1, left), (1, right)), extrapolate=False)


def q_axis_and_minimum(psi, q):
    """Local axis estimate and sampled minimum, including the estimated axis."""
    q0 = float(q[0] - psi[0] * (q[1] - q[0]) / (psi[1] - psi[0]))
    index = int(np.argmin(q))
    # Only roundoff equality is collapsed. No physical reversed-shear threshold.
    if index == 0 or q[index] >= q0 - 1e-12 * max(1., abs(q0)):
        return q0, q0, 0.
    return q0, float(q[index]), float(psi[index])


def load_equilibrium(case_dir):
    """Return 13 scalars, supporting profiles and extraction metadata."""
    case_dir = Path(case_dir)
    with netcdf_file(case_dir / 'mapdsk.cdf', mmap=False) as f:
        d = {k: f.variables[k].data.copy() for k in
             ('qa', 'pa', 'PsiBig', 'X0', 'Xmag', 'Zmag', 'xinf', 'zinf', 'imach')}
    if not all(np.all(np.isfinite(v)) for v in d.values()):
        raise ValueError('Nonfinite mapped equilibrium values')
    geom = geometry(d['xinf'], d['zinf'], float(d['Zmag']))
    psi = d['PsiBig'] / d['PsiBig'][-1]
    q, p = d['qa'], d['pa']
    if len(q) < 4 or q.shape != psi.shape or p.shape != psi.shape or not np.all(np.diff(psi) > 0):
        raise ValueError('Invalid mapped radial profile grid')
    if not np.all(q > 0) or not np.all(p >= 0):
        raise ValueError('Expected positive q and nonnegative mapped pressure')
    # Verify the mapped copies rather than selecting one of the outequ q profiles.
    n = len(q)
    raw = np.fromfile(case_dir / 'mpout1', dtype='<f8', count=49 + 4*n)
    qp = np.loadtxt(case_dir / 'qpvspsi.dat', skiprows=1, max_rows=n)
    if (raw.size != 49 + 4*n or not np.array_equal(q, raw[49+2*n:49+3*n])
            or not np.array_equal(p, raw[49:49+n])
            or not np.array_equal(qp, np.column_stack((psi, q, p)))):
        raise ValueError('Mapped qa/pa/psi copies disagree')
    blocks, headers = read_transp(case_dir / 'transp.dat')
    for key in SPECIES:
        if key not in headers:
            raise ValueError(f'Missing TRANSP species block {key}')
        if 'no-such-item' in headers[key] and np.any(blocks[key] != 0):
            raise ValueError(f'Nonzero no-such-item payload: {key}')
    if 'TESLA*CM' not in headers.get('BZXR', ''):
        raise ValueError('Missing BZXR or unexpected unit (expected TESLA*CM)')
    density_psi, rho = processed_density(blocks)
    normalized = rho / rho[0]
    spline = density_spline(density_psi, normalized)
    bv = abs(float(blocks['BZXR'][0])) / (100 * geom['R_v_m'])
    q0, qmin, psi_qmin = q_axis_and_minimum(psi, q)
    params = dict(a=geom['a_m'], R_v=geom['R_v_m'], delta=geom['delta'],
                  kappa=geom['kappa'], B_v=bv, q_0=q0, q_lcfs=float(q[-1]),
                  q_min=qmin, p_axis=float(p[0])*bv**2/MU0, rho_axis=float(rho[0]),
                  delta_axis=(float(d['Xmag'])-geom['R_v_m'])/geom['a_m'],
                  psi_N_qmin=psi_qmin, rho_half_over_axis=float(spline(.5)))
    if not all(np.isfinite(v) for v in params.values()) or bv <= 0:
        raise ValueError('Invalid equilibrium scalar values')
    profile_r = np.linspace(0, 1, n)
    rho_profile = spline(profile_r**2)
    if not np.all(rho_profile > 0):
        raise ValueError('Reconstructed NOVA density spline is nonpositive')
    meta = dict(R_axis=float(d['Xmag']), Z_axis=float(d['Zmag']), imach=int(d['imach']),
                q_first_mapped=float(q[0]), psi_N_first_mapped=float(psi[0]),
                p_axis_method='first_mapped_sample_near_axis',
                q_axis_method='linear_extrapolation_in_psi_N_first_two_mapped_points',
                density_method='NOVA_iden4_TRANSP_species_smooth_extrap1_depose',
                mu0_SI=MU0, proton_mass_kg=proton_mass,
                V_A_ref_m_s=bv/np.sqrt(MU0*rho[0]),
                delta_upper=geom['delta_upper'], delta_lower=geom['delta_lower'])
    meta['omega_ref_rad_s'] = meta['V_A_ref_m_s'] / (params['q_lcfs']*params['R_v'])
    profiles = dict(psi_N_mapped=psi, q=q, p_Pa=p*bv**2/MU0,
                    r=profile_r, psi_N=profile_r**2, rho_over_axis=rho_profile,
                    density_psi_N_knots=density_psi, density_rho_over_axis_knots=normalized)
    return params, profiles, meta


