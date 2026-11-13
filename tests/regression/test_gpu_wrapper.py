"""
wrap_gpu_mf without a GPU.

cupy and gpu4pyscf are replaced by CPU stand-ins (numpy, and a PySCF mean-field object whose
arrays have a .get() method), so the wrapper's own logic is exercised: the complex-density
algebra for restricted and unrestricted references, and that the "GPU" object follows the
nuclei in RT_Ehrenfest. This does not test gpu4pyscf itself.
"""
import os
import sys
import types
import numpy as np
import pytest
from pyscf import gto, dft
from tides import RT_Ehrenfest
from tides.utils.rt_gpu import wrap_gpu_mf


class _Arr(np.ndarray):
    def get(self):
        return np.asarray(self)


def _as_gpu(a):
    return np.asarray(a).view(_Arr)


@pytest.fixture
def fake_gpu(monkeypatch):
    cupy = types.ModuleType('cupy')
    cupy.asarray = _as_gpu
    cupy.stack = lambda seq: _as_gpu(np.stack(seq))
    cupy.einsum = np.einsum
    gpu4pyscf = types.ModuleType('gpu4pyscf')
    gpu4pyscf.dft = types.ModuleType('gpu4pyscf.dft')
    monkeypatch.setitem(sys.modules, 'cupy', cupy)
    monkeypatch.setitem(sys.modules, 'gpu4pyscf', gpu4pyscf)
    monkeypatch.setitem(sys.modules, 'gpu4pyscf.dft', gpu4pyscf.dft)


def _mf(kind, xc, atom='Li 0 0 0; H 0 0 1.7', spin=0, charge=0, gpu_like=False):
    mol = gto.M(verbose=0, atom=atom, basis='6-31G', spin=spin, charge=charge)
    mf = (dft.UKS if kind == 'UKS' else dft.RKS)(mol).density_fit()
    mf.xc = xc
    mf.conv_tol = 1e-11
    mf.kernel()
    if gpu_like:     # what the wrapper expects from a gpu4pyscf object
        mf.mo_coeff, mf.mo_energy, mf.mo_occ = _as_gpu(mf.mo_coeff), _as_gpu(mf.mo_energy), _as_gpu(mf.mo_occ)
    return mf


def _complex_dm(mf, theta=0.4):
    C = np.array(mf.mo_coeff, dtype=complex)
    def rot(c, nocc):
        c = c.copy(); h, l = nocc - 1, nocc
        ch, cl = c[:, h].copy(), c[:, l].copy()
        c[:, h] = np.cos(theta) * ch + 1j * np.sin(theta) * cl
        c[:, l] = 1j * np.sin(theta) * ch + np.cos(theta) * cl
        return c
    occ = np.asarray(mf.mo_occ)
    if C.ndim == 2:
        C = rot(C, int(round(occ.sum() / 2)))
    else:
        C = np.stack([rot(C[0], int(occ[0].sum())), rot(C[1], int(occ[1].sum()))])
    return mf.make_rdm1(C, occ)


@pytest.mark.parametrize('kind,xc,spin,charge', [
    ('UKS', 'PBE0', 0, 0), ('UKS', 'CAMB3LYP', 1, 1), ('RKS', 'PBE0', 0, 0), ('RKS', 'CAMB3LYP', 0, 0), ('RKS', 'PBE', 0, 0)])
def test_complex_density_fock_and_energy_match_cpu(fake_gpu, kind, xc, spin, charge):
    ref = _mf(kind, xc, spin=spin, charge=charge)
    wrapped = wrap_gpu_mf(_mf(kind, xc, spin=spin, charge=charge, gpu_like=True))
    dm = _complex_dm(ref)
    assert np.linalg.norm(dm.imag) > 0.1
    # Without exact exchange the antisymmetric imaginary part contributes nothing, and PySCF's CPU
    # density-fitted J refuses complex input, so the reference is the real part alone.
    dm_ref = dm if xc != 'PBE' else dm.real
    assert np.allclose(wrapped.get_fock(dm=dm), ref.get_fock(dm=dm_ref), atol=1e-8)
    assert abs(wrapped.energy_tot(dm=dm) - ref.energy_tot(dm=dm_ref)) < 1e-8


@pytest.mark.parametrize('kind', ['UKS', 'RKS'])
def test_wrapper_follows_nuclei_in_ehrenfest(fake_gpu, kind):
    def run(mf):
        rt = RT_Ehrenfest(mf, 0.05, 1.0, filename=os.devnull, prop='etrs', chkfile=None, Ne_step=2, N_step=2)
        rt.observables.update(mo_occ=True)      # re-solves the SCF at every print: must go through the wrapper too
        rt.nuc.vel[0, 2], rt.nuc.vel[1, 2] = -2e-3, 2e-3
        e0 = rt._scf.energy_tot(dm=rt.den_ao) + np.sum(rt.nuc.get_ke())
        rt.kernel()
        e1 = rt._scf.energy_tot(dm=rt.den_ao) + np.sum(rt.nuc.get_ke())
        return rt, e1 - e0

    atom = 'H 0 0 0; H 0 0 0.9'
    gpu_obj = _mf(kind, 'PBE0', atom=atom, gpu_like=True)
    rt_w, drift_w = run(wrap_gpu_mf(gpu_obj))
    rt_c, drift_c = run(_mf(kind, 'PBE0', atom=atom))

    # the "GPU" object sits at the final geometry, not the initial one
    assert np.allclose(gpu_obj.mol.atom_coords(), rt_w.nuc.pos, atol=1e-12)
    assert abs(rt_w.nuc.pos[1, 2] - rt_w.nuc.pos[0, 2] - 0.9 / 0.529177210903) > 1e-3
    # same trajectory and density as the plain CPU calculation, and energy is conserved
    assert np.allclose(rt_w.nuc.pos, rt_c.nuc.pos, atol=1e-8)
    assert np.allclose(rt_w.den_ao, rt_c.den_ao, atol=1e-6)
    assert np.allclose(rt_w._mo_occ, rt_c._mo_occ, atol=1e-6)
    assert abs(drift_w) < 1e-5
    assert abs(drift_w - drift_c) < 1e-7
