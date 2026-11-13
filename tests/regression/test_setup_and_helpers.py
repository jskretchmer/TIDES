from pathlib import Path
import numpy as np
import pytest
from pyscf import gto, scf
from tides import RT_SCF
from tides.utils.rt_utils import single_excite
from rt_helpers import water, NULL

REPO = Path(__file__).resolve().parents[2]


def test_all_subpackages_are_installed():
    setupcfg = pytest.importorskip('setuptools.config.setupcfg')
    packages = setupcfg.read_configuration(str(REPO / 'setup.cfg'))['options']['packages']
    on_disk = {'.'.join(p.parent.relative_to(REPO / 'src').parts) for p in (REPO / 'src' / 'tides').rglob('__init__.py')}
    assert on_disk and on_disk <= set(packages)


def test_spin_square_observable_runs():
    mol = gto.M(verbose=0, atom='O 0 0 0; H 0 0 0.97', basis='sto-3g', spin=1)
    mf = scf.UHF(mol)
    mf.kernel()
    rt = RT_SCF(mf, 0.2, 0.4, filename=NULL, chkfile=None)
    rt.observables.update(spin_square=True)
    rt.kernel()
    assert abs(rt._s2 - 0.75) < 0.05 and abs(rt._2s_p1 - 2) < 0.05


def test_single_excite_on_restricted_reference():
    rt = RT_SCF(water(), 0.2, 0.4, filename=NULL, chkfile=None)
    single_excite(rt, excitation_alpha_from=5, excitation_alpha_to=6)
    assert list(rt.occ) == [2, 2, 2, 2, 1, 1, 0]
    assert abs(np.trace(rt.den_ao @ rt.ovlp).real - 10) < 1e-10
    with pytest.raises(ValueError):
        single_excite(rt, excitation_alpha_from=7)             # empty orbital


def test_occupations_come_from_the_scf_object():
    mol = gto.M(verbose=0, atom='O 0 0 0; H 0 0 0.97', basis='6-31G', spin=1)
    mf = scf.UHF(mol)
    mf.kernel()
    occ = mf.mo_occ.copy()
    occ[0][3], occ[0][5] = 0, 1                                 # a non-Aufbau (Delta-SCF-like) alpha occupation
    mf.mo_occ = occ
    rt = RT_SCF(mf, 0.2, 0.4, filename=NULL, chkfile=None)
    assert np.array_equal(rt.occ, occ)
    rt.occ[0][0] = 0                                            # editing rt.occ must not touch the SCF object
    assert mf.mo_occ[0][0] == 1


def test_reassigning_orth_keeps_the_ground_state_stationary():
    mf = water()
    rt = RT_SCF(mf, 0.2, 2.0, filename=NULL, chkfile=None)
    P0 = rt.den_ao.copy()
    e, v = np.linalg.eigh(rt.ovlp)
    rt.orth = (v / np.sqrt(e)) @ v.T                            # switch to symmetric orthogonalisation
    assert np.allclose(rt.orth_inv @ rt.orth, np.eye(len(e)), atol=1e-12)
    rt.kernel()
    assert np.linalg.norm(rt.den_ao - P0) < 1e-6


def test_linearly_dependent_basis_gives_a_clear_error():
    mol = gto.M(verbose=0, atom='He 0 0 0; He 0 0 0.03', basis='aug-cc-pvtz')
    assert np.linalg.eigvalsh(mol.intor('int1e_ovlp')).min() < 1e-7
    mf = scf.RHF(mol)
    mf.max_cycle = 1
    mf.kernel()
    with pytest.raises(ValueError, match='linearly dependent'):
        RT_SCF(mf, 0.05, 0.1, filename=NULL, chkfile=None)
