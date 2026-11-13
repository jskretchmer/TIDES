import numpy as np
import pytest
from pyscf import gto, scf
from tides import RT_SCF, RT_Ehrenfest
from rt_helpers import water, rotated, NULL


@pytest.mark.parametrize('kind', ['RHF', 'UHF'])
@pytest.mark.parametrize('prop,tol', [('magnus_interpol', 1e-10), ('magnus_step', 1e-10), ('rk4', 1e-10),
                                      ('etrs', 1e-6), ('ep_pc', 1e-6), ('cfm4', 1e-6)])
def test_restart_reproduces_continuous_run(tmp_path, kind, prop, tol):
    chk = str(tmp_path / 'run.chk')
    dt = 0.01 if prop == 'rk4' else 0.1            # rk4 needs dt * max|eps| < 2.8
    T1, T2 = 50 * dt, 100 * dt

    def job(T, chkfile):
        mf = water(kind)
        mf.mo_coeff = rotated(mf)
        rt = RT_SCF(mf, dt, T, filename=NULL, prop=prop, chkfile=chkfile)
        rt.kernel()
        return rt.den_ao

    P_continuous = job(T2, None)
    job(T1, chk)
    P_restarted = job(T2, chk)                      # an identical input, picking up the checkpoint
    assert np.linalg.norm(P_restarted - P_continuous) < tol


@pytest.mark.parametrize('kind', ['RHF', 'UHF'])
@pytest.mark.parametrize('prop', ['magnus_interpol', 'etrs', 'ep_pc'])
def test_ehrenfest_restart_reproduces_continuous_run(tmp_path, kind, prop):
    chk = str(tmp_path / 'eh.chk')

    def job(T, chkfile, kick):
        mol = gto.M(verbose=0, atom='H 0 0 0; H 0 0 0.9', basis='6-31G')
        mf = (scf.UHF if kind == 'UHF' else scf.RHF)(mol)
        mf.kernel()
        rt = RT_Ehrenfest(mf, 0.1, T, filename=NULL, prop=prop, chkfile=chkfile, Ne_step=2, N_step=2)
        if kick:
            rt.nuc.vel[0, 2], rt.nuc.vel[1, 2] = -1e-3, 1e-3
        rt.kernel()
        return rt

    cont = job(3.2, None, True)
    job(1.6, chk, True)
    rest = job(3.2, chk, False)
    assert np.allclose(rest.nuc.pos, cont.nuc.pos, atol=1e-9)
    assert np.allclose(rest.nuc.vel, cont.nuc.vel, atol=1e-9)
    assert np.linalg.norm(rest.den_ao - cont.den_ao) < 1e-6


def test_restart_with_nothing_left_to_do_warns(tmp_path):
    chk = str(tmp_path / 'tides.chk')
    mf = water()
    mf.mo_coeff = rotated(mf)
    RT_SCF(mf, 0.2, 1.0, filename=NULL, chkfile=chk).kernel()
    # a new calculation that happens to find the old checkpoint
    with pytest.warns(UserWarning, match='no steps will be taken'):
        RT_SCF(water(), 0.2, 1.0, filename=NULL, chkfile=chk)


def test_checkpoint_from_another_system_is_refused(tmp_path):
    chk = str(tmp_path / 'tides.chk')
    RT_SCF(water(), 0.2, 0.4, filename=NULL, chkfile=chk).kernel()
    with pytest.raises(ValueError, match='different calculation'):
        RT_SCF(water(basis='6-31G'), 0.2, 2.0, filename=NULL, chkfile=chk)
