import numpy as np
import pytest
from tides import RT_SCF, MOCAP
from rt_helpers import water, rotated, nelec, record, NULL


def _run(mf, C0, prop, dt, T, cap=False, tight=True):
    mf.mo_coeff = C0.copy()
    rt = RT_SCF(mf, dt, T, filename=NULL, prop=prop, chkfile=None)
    if tight:
        rt.magnus_tolerance, rt.magnus_maxiter = 1e-13, 300
    if cap:
        rt.add_potential(MOCAP(1.0, 0.5, 1.0, 100.0))
    trace = record(rt, 'n', nelec)
    rt.kernel()
    return rt, np.array(trace)


@pytest.mark.parametrize('kind', ['RHF', 'UHF'])
@pytest.mark.parametrize('prop', ['magnus_interpol', 'etrs', 'ep_pc', 'cfm4', 'magnus_step'])
def test_unitary_integrators_keep_norm_idempotency_hermiticity(kind, prop):
    mf = water(kind)
    rt, n = _run(mf, rotated(mf), prop, 0.1, 2.0, tight=False)
    S = rt.ovlp
    assert np.abs(n - n[0]).max() < 1e-10
    dms = [rt.den_ao / 2] if kind == 'RHF' else list(rt.den_ao)
    for P in dms:
        assert np.linalg.norm(P - P.conj().T) < 1e-12
        assert np.linalg.norm(P @ S @ P - P) < 1e-9


@pytest.mark.parametrize('prop', ['magnus_interpol', 'etrs', 'ep_pc', 'magnus_step'])
def test_second_order_convergence(prop):
    mf = water()
    C0 = rotated(mf)
    ref, _ = _run(mf, C0, 'etrs', 0.005, 2.0)
    err = [np.linalg.norm(_run(mf, C0, prop, dt, 2.0)[0].den_ao - ref.den_ao) for dt in (0.05, 0.025)]
    assert err[0] / err[1] > 3.0        # 4 for exact second order; magnus_step gave 2 before its start-up fix


@pytest.mark.parametrize('prop', ['magnus_interpol', 'etrs', 'ep_pc', 'cfm4'])
def test_cap_only_removes_norm(prop):
    mf = water()
    _, n = _run(mf, rotated(mf, theta=0.5, top=True), prop, 0.05, 2.0, cap=True, tight=False)
    assert n[-1] < n[0] - 1e-2
    assert np.diff(n).max() <= 1e-12


def test_rk4_does_not_hide_norm_changes():
    mf = water()
    C0 = rotated(mf, theta=0.5, top=True)
    # with a CAP, rk4 must absorb what the unitary integrators absorb (a re-orthogonalisation would give exactly 10)
    _, n_ref = _run(mf, C0, 'etrs', 0.005, 0.5, cap=True)
    _, n_rk4 = _run(mf, C0, 'rk4', 0.005, 0.5, cap=True)
    assert n_ref[-1] < n_ref[0] - 1e-3
    assert abs(n_rk4[-1] - n_ref[-1]) < 1e-4
    # beyond the stability limit the failure has to show up in the electron count
    _, n_bad = _run(mf, C0, 'rk4', 0.1, 2.0)
    assert not abs(n_bad[-1] - n_bad[0]) < 1e-3
