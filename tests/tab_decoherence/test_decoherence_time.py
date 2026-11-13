import numpy as np
import pytest

from tides.decoherence import (
    time_averaged_force,
    decoherence_time,
    decoherence_time_matrix,
)

# Increment 1 of the rTAB-w1 build (see .claude/skills/tab-decoherence).
# Equations verified against the typeset PDFs:
#   F^avg  : J. Chem. Phys. 155, 214101 (2021), Eq. 2
#   tau_ij : J. Chem. Phys. 155, 214101 (2021), Eq. 1 (factor of 8 -> Paper C)


# ---- time-averaged force (Eq. 2) -------------------------------------------

def test_time_averaged_force_single_vector():
    f0 = np.array([2.0, -1.0])
    fdt = np.array([4.0, 1.0])
    np.testing.assert_allclose(time_averaged_force(f0, fdt), [3.0, 0.0])


def test_time_averaged_force_per_state_stack():
    f0 = np.array([[1.0], [3.0]])
    fdt = np.array([[3.0], [5.0]])
    np.testing.assert_allclose(time_averaged_force(f0, fdt), [[2.0], [4.0]])


# ---- tau_ij hand values (Eq. 1) --------------------------------------------

def test_tau_hand_value_single_dof():
    # hbar=1, alpha=1, dF=2  ->  1/tau^2 = 4 / (8*1*1) = 0.5  ->  tau = sqrt(2)
    tau = decoherence_time([0.0], [2.0], alpha=1.0, hbar=1.0)
    assert tau == pytest.approx(np.sqrt(2.0))


def test_tau_closed_form_single_dof():
    # For one DOF, tau = sqrt(8 hbar^2 alpha) / |dF|
    dF, alpha, hbar = 3.0, 2.5, 1.3
    tau = decoherence_time([0.0], [dF], alpha=alpha, hbar=hbar)
    assert tau == pytest.approx(np.sqrt(8.0 * hbar ** 2 * alpha) / abs(dF))


def test_tau_multi_dof_adds_in_inverse_square():
    favg_i = np.array([0.0, 0.0])
    favg_j = np.array([2.0, 4.0])
    alpha = np.array([1.0, 2.0])
    inv_tau2 = (2.0 ** 2) / (8 * 1 * 1.0) + (4.0 ** 2) / (8 * 1 * 2.0)
    assert decoherence_time(favg_i, favg_j, alpha) == pytest.approx(1.0 / np.sqrt(inv_tau2))


# ---- structural properties -------------------------------------------------

def test_parallel_pes_gives_infinite_tau():
    # states 0 and 1 have identical forces (parallel PESs) -> no decoherence
    favg = np.array([[1.0, 2.0], [1.0, 2.0], [5.0, -3.0]])
    tau = decoherence_time_matrix(favg, alpha=1.0)
    assert np.isinf(tau[0, 1]) and np.isinf(tau[1, 0])
    assert np.isfinite(tau[0, 2])
    assert np.isfinite(tau[1, 2])


def test_diagonal_is_infinite():
    favg = np.array([[1.0], [4.0], [-2.0]])
    tau = decoherence_time_matrix(favg, alpha=1.0)
    assert np.all(np.isinf(np.diag(tau)))


def test_matrix_is_symmetric():
    rng = np.random.default_rng(0)
    favg = rng.normal(size=(5, 3))
    tau = decoherence_time_matrix(favg, alpha=np.array([1.0, 2.0, 0.5]))
    np.testing.assert_allclose(tau, tau.T)


def test_matrix_matches_scalar_wrapper():
    favg = np.array([[0.0, 1.0], [2.0, -1.0], [3.0, 3.0]])
    alpha = np.array([1.5, 0.75])
    tau = decoherence_time_matrix(favg, alpha=alpha, hbar=1.1)
    for i in range(3):
        for j in range(3):
            if i != j:
                assert decoherence_time(favg[i], favg[j], alpha, hbar=1.1) == pytest.approx(tau[i, j])


# ---- parameter handling ----------------------------------------------------

def test_scalar_alpha_matches_broadcast_vector():
    favg = np.array([[0.0, 0.0], [1.0, 1.0]])
    t_scalar = decoherence_time_matrix(favg, alpha=2.0)
    t_vector = decoherence_time_matrix(favg, alpha=np.array([2.0, 2.0]))
    np.testing.assert_allclose(t_scalar, t_vector)


def test_tau_scales_as_sqrt_alpha():
    favg = np.array([[0.0], [1.0]])
    t1 = decoherence_time_matrix(favg, alpha=1.0)[0, 1]
    t4 = decoherence_time_matrix(favg, alpha=4.0)[0, 1]
    assert t4 == pytest.approx(2.0 * t1)


def test_tau_scales_linearly_with_hbar():
    favg = np.array([[0.0], [1.0]])
    t1 = decoherence_time_matrix(favg, alpha=1.0, hbar=1.0)[0, 1]
    t2 = decoherence_time_matrix(favg, alpha=1.0, hbar=2.0)[0, 1]
    assert t2 == pytest.approx(2.0 * t1)


def test_nonpositive_alpha_raises():
    favg = np.array([[0.0], [1.0]])
    with pytest.raises(ValueError):
        decoherence_time_matrix(favg, alpha=0.0)
    with pytest.raises(ValueError):
        decoherence_time_matrix(favg, alpha=-1.0)


def test_wrong_favg_ndim_raises():
    with pytest.raises(ValueError):
        decoherence_time_matrix(np.array([1.0, 2.0, 3.0]), alpha=1.0)
