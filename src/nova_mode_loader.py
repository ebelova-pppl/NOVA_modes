import numpy as np
import os
import warnings
from pathlib import Path

# read NOVA AE mode structures from NOVA output files as egn02w.7439E+00
# in /u/ebelova/NOVA/nstx_120113/ on Flux or
# in /global/cfs/cdirs/m314/nova/nstx_120113 on Perlmutter

def load_mode_from_nova(path):
    """
    It returns a 2D numpy array: mode[m_index, r_index]

    Here f1 is 1D array of size 3*nr*nhar + 4,
    it contains 3 perturbations: xi_psi, delta_p, xi_surf,
    and 4 scalar parameters: 
    f1[0]= omega, 
    f1[-3]= nr (=number of radial points),
    f1[-2]= gamma_d of continuum decay, 
    f1[-1]= ntor

    """
    f1 = np.fromfile(path)

    omega = float(f1[0])
    nr = int(f1[-3])
    gamma_d = float(f1[-2])
    ntor = int(round(float(f1[-1])))

    payload = f1.size - 4
    denom = 3 * nr
    if payload % denom != 0:
        raise ValueError(
            f"Bad NOVA file size for {path}: (size-4)={payload} not divisible by 3*nr={denom}"
        )
    nhar = payload // denom        # number of poloidal harmonics

    f11 = f1[1:-3].reshape(3, nhar, nr)
    mode = f11[0, :, :]  # already (nhar, nr)

    return mode, omega, gamma_d, ntor


