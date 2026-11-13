"""Small shared helpers for the regression tests (tiny systems, a few seconds each)."""
import os
import numpy as np
from pyscf import gto, scf

WATER = 'O 0 0 -0.348; H 0 0.760 -0.933; H 0 -0.760 -0.933'
NULL = os.devnull


def water(kind='RHF', basis='sto-3g'):
    spin, charge = (1, 1) if kind == 'UHF' else (0, 0)
    mol = gto.M(verbose=0, atom=WATER, basis=basis, spin=spin, charge=charge)
    mf = (scf.UHF if kind == 'UHF' else scf.RHF)(mol)
    mf.conv_tol = 1e-11
    mf.kernel()
    return mf


def rotated(mf, theta=0.3, imag=False, top=False):
    """Non-stationary start: mix the HOMO with the LUMO (or the highest virtual). Columns stay orthonormal."""
    C = np.array(mf.mo_coeff, dtype=complex)
    f = 1j if imag else 1.0

    def rot(c, nocc):
        c = c.copy()
        h, l = nocc - 1, (c.shape[1] - 1 if top else nocc)
        ch, cl = c[:, h].copy(), c[:, l].copy()
        c[:, h] = np.cos(theta) * ch + f * np.sin(theta) * cl
        c[:, l] = -np.conj(f) * np.sin(theta) * ch + np.cos(theta) * cl
        return c

    occ = np.asarray(mf.mo_occ)
    if C.ndim == 2:
        return rot(C, int(round(occ.sum() / 2)))
    return np.stack([rot(C[0], int(occ[0].sum())), rot(C[1], int(occ[1].sum()))])


def nelec(rt):
    P, S = rt.den_ao, rt.ovlp
    if P.ndim == 2:
        return np.trace(P @ S).real
    return sum(np.trace(p @ S).real for p in P)


def record(rt, name, fn):
    """Collect fn(rt) at every print through a custom observable."""
    out = []
    rt._observables_functions[name] = [lambda r: out.append(fn(r)), lambda r: None]
    rt.observables[name] = True
    return out
