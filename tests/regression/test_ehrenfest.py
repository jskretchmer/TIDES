import numpy as np
import pytest
from pyscf import gto, scf
from tides import RT_Ehrenfest
from tides.nuclear import ehrenfest_force
from tides.utils.rt_utils import excite
from rt_helpers import rotated, nelec, record, NULL

AMU = 1822.888486209


def _rt(atom='Li 0 0 0; H 0 0 1.7', kind='RHF', basis='6-31G', T=1.0, dt=0.05, **kw):
    mol = gto.M(verbose=0, atom=atom, basis=basis, **{k: kw.pop(k) for k in ('ecp', 'spin', 'charge') if k in kw})
    mf = (scf.UHF if kind == 'UHF' else scf.RHF)(mol)
    mf.conv_tol = 1e-12
    mf.kernel()
    kw.setdefault('Ne_step', 1)
    kw.setdefault('N_step', 1)
    return RT_Ehrenfest(mf, dt, T, filename=NULL, chkfile=None, **kw)


def _complexify(rt):
    rt._scf.mo_coeff = rotated(rt._scf, theta=0.4, imag=True)
    rt.den_ao = rt._scf.make_rdm1(mo_occ=rt.occ)


def _lowdin(S):
    e, v = np.linalg.eigh(S)
    return (v / np.sqrt(e)) @ v.T, (v * np.sqrt(e)) @ v.T


@pytest.mark.parametrize('kind', ['RHF', 'UHF'])
def test_force_matches_finite_difference_for_complex_density(kind):
    """-dE/dR at fixed density in the Lowdin basis, which is what the propagation holds fixed."""
    rt = _rt(kind=kind)
    _complexify(rt)
    assert np.linalg.norm(np.imag(rt.den_ao)) > 0.1
    F = ehrenfest_force.get_force(rt)

    pos0 = rt.nuc.pos.copy()
    _, Xinv = _lowdin(rt._scf.get_ovlp())
    stack = lambda f, P: f(P) if P.ndim == 2 else np.stack([f(p) for p in P])
    P_orth = stack(lambda p: Xinv @ p @ Xinv, rt.den_ao)

    def energy(dz):
        rt.nuc.pos = pos0.copy()
        rt.nuc.pos[1, 2] += dz
        rt.update_mol()
        X, _ = _lowdin(rt._scf.get_ovlp())
        return rt._scf.energy_tot(dm=stack(lambda p: X @ p @ X, P_orth))

    h = 1e-4
    assert abs(F[1, 2] + (energy(h) - energy(-h)) / (2 * h)) < 1e-7


def test_force_is_recomputed_for_the_prepared_state():
    rt = _rt(kind='UHF', T=0.0)
    stale = rt.nuc.force.copy()
    excite(rt, excitation_alpha=2)                  # remove the alpha HOMO electron
    rt.kernel()                                     # zero steps: only the start-up
    assert np.allclose(rt.nuc.force, ehrenfest_force.get_force(rt), atol=1e-10)
    assert np.abs(rt.nuc.force - stale).max() > 1e-2


def test_ecp_and_masses_survive_the_geometry_update():
    rt = _rt(atom='I 0 0 0; H 0 0 1.61', basis='lanl2dz', ecp='lanl2dz', T=0.1)
    e_scf = rt._scf.e_tot
    assert rt._scf.mol.has_ecp()
    assert rt._scf.mol.nelectron == int(np.sum(rt.occ)) == 8
    assert abs(rt._scf.energy_tot(dm=rt.den_ao) - e_scf) < 1e-8
    assert abs(rt.nuc.mass[0, 0] / AMU - 126.904) < 1e-2     # iodine, not the element with Z = 53 - 46
    assert abs(rt.nuc.mass[1, 0] / AMU - 1.008) < 1e-2


def test_ghost_atom_stays_put_and_does_not_produce_nan():
    rt = _rt(atom='H 0 0 0; H 0 0 0.74; GHOST-He 0 0 3.0', T=0.2)
    ghost0 = rt.nuc.pos[2].copy()
    rt.nuc.vel[0, 2], rt.nuc.vel[1, 2] = -1e-3, 1e-3
    rt.kernel()
    assert np.all(np.isfinite(rt.nuc.pos)) and np.all(np.isfinite(rt.nuc.vel))
    assert np.array_equal(rt.nuc.pos[2], ghost0)
    assert np.isfinite(np.sum(rt.nuc.get_ke()))


def test_no_scf_is_run_at_prints_unless_mo_occupations_are_requested():
    rt = _rt(atom='H 0 0 0; H 0 0 0.9', T=0.5)
    calls = [0]
    orig = rt._scf.kernel

    def counting(*a, **k):
        calls[0] += 1
        return orig(*a, **k)

    rt._scf.kernel = counting
    rt.observables.update(energy=True)
    rt.kernel()
    assert calls[0] == 0


@pytest.mark.parametrize('prop,tol', [('magnus_interpol', 2e-6), ('etrs', 2e-5), ('magnus_step', 2e-4)])
def test_energy_and_norm_conservation_with_moving_nuclei(prop, tol):
    rt = _rt(atom='H 0 0 0; H 0 0 0.9', T=10.0, prop=prop)
    _complexify(rt)
    rt.nuc.vel[0, 2], rt.nuc.vel[1, 2] = -2e-3, 2e-3
    E = record(rt, 'e', lambda r: r._scf.energy_tot(dm=r.den_ao) + np.sum(r.nuc.get_ke()))
    N = record(rt, 'n', nelec)
    rt.kernel()
    E, N = np.array(E), np.array(N)
    assert abs(rt.nuc.pos[1, 2] - rt.nuc.pos[0, 2] - 0.9 / 0.529177210903) > 1e-2     # the bond really moved
    assert np.abs(N - 2).max() < 1e-9
    assert np.abs(E - E[0]).max() < tol
