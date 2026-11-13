import numpy as np
import pytest

from tides.decoherence import (
    exponential_target,
    build_exponential_target,
    decoherence_time_matrix,
)

# Increment 2 of the rTAB-w1 build (see .claude/skills/tab-decoherence).
# Equations verified against the typeset PDF:
#   rho^d_ii = rho^c_ii               J. Chem. Phys. 155, 214101 (2021), Eq. 3
#   rho^d_ij = rho^c_ij exp(-dt/tau)  J. Chem. Phys. 155, 214101 (2021), Eq. 4


def _hermitian_rho(nstates, seed=0):
    '''A random Hermitian, unit-trace density-matrix-like array for testing.'''
    rng = np.random.default_rng(seed)
    a = rng.normal(size=(nstates, nstates)) + 1j * rng.normal(size=(nstates, nstates))
    rho = a @ a.conj().T
    return rho / np.trace(rho).real


# ---- diagonal held fixed (Eq. 3) -------------------------------------------

def test_diagonal_unchanged():
    rho_c = _hermitian_rho(4)
    tau = decoherence_time_matrix(np.arange(4.0)[:, None], alpha=1.0)
    rho_d = exponential_target(rho_c, tau, dt=0.5)
    np.testing.assert_allclose(np.diagonal(rho_d), np.diagonal(rho_c))


def test_diagonal_fixed_even_if_tau_diagonal_finite():
    # Contract: diagonal follows rho_c regardless of what tau's diagonal holds.
    rho_c = _hermitian_rho(3)
    tau = np.full((3, 3), 2.0)  # finite diagonal on purpose
    rho_d = exponential_target(rho_c, tau, dt=0.7)
    np.testing.assert_allclose(np.diagonal(rho_d), np.diagonal(rho_c))


# ---- off-diagonal decay (Eq. 4) --------------------------------------------

def test_offdiagonal_decays_by_exp_factor():
    rho_c = _hermitian_rho(3, seed=2)
    tau = np.array([[np.inf, 1.0, 4.0],
                    [1.0, np.inf, 2.0],
                    [4.0, 2.0, np.inf]])
    dt = 0.5
    rho_d = exponential_target(rho_c, tau, dt)
    for i in range(3):
        for j in range(3):
            if i != j:
                assert rho_d[i, j] == pytest.approx(rho_c[i, j] * np.exp(-dt / tau[i, j]))


def test_parallel_or_infinite_tau_leaves_coherence_intact():
    rho_c = _hermitian_rho(2, seed=3)
    tau = np.array([[np.inf, np.inf], [np.inf, np.inf]])
    rho_d = exponential_target(rho_c, tau, dt=1.0)
    np.testing.assert_allclose(rho_d, rho_c)


def test_tiny_tau_kills_coherence():
    rho_c = _hermitian_rho(2, seed=4)
    tau = np.array([[np.inf, 1e-12], [1e-12, np.inf]])
    rho_d = exponential_target(rho_c, tau, dt=1.0)
    assert rho_d[0, 1] == pytest.approx(0.0)
    assert rho_d[1, 0] == pytest.approx(0.0)


def test_zero_timestep_is_identity():
    rho_c = _hermitian_rho(4, seed=5)
    tau = decoherence_time_matrix(np.arange(4.0)[:, None], alpha=1.0)
    rho_d = exponential_target(rho_c, tau, dt=0.0)
    np.testing.assert_allclose(rho_d, rho_c)


# ---- structural guarantees -------------------------------------------------

def test_preserves_hermiticity():
    rho_c = _hermitian_rho(5, seed=6)
    tau = decoherence_time_matrix(np.linspace(-1, 1, 5)[:, None], alpha=0.8)
    rho_d = exponential_target(rho_c, tau, dt=0.3)
    np.testing.assert_allclose(rho_d, rho_d.conj().T)


def test_preserves_trace_population():
    rho_c = _hermitian_rho(5, seed=7)
    tau = decoherence_time_matrix(np.linspace(-2, 2, 5)[:, None], alpha=1.2)
    rho_d = exponential_target(rho_c, tau, dt=0.9)
    assert np.trace(rho_d) == pytest.approx(np.trace(rho_c))


def test_phase_of_coherence_preserved_only_magnitude_scaled():
    rho_c = _hermitian_rho(3, seed=8)
    tau = np.array([[np.inf, 1.5, 3.0],
                    [1.5, np.inf, 2.5],
                    [3.0, 2.5, np.inf]])
    rho_d = exponential_target(rho_c, tau, dt=0.4)
    for i in range(3):
        for j in range(3):
            if i != j and abs(rho_c[i, j]) > 0:
                # real positive decay factor -> identical complex phase
                assert np.angle(rho_d[i, j]) == pytest.approx(np.angle(rho_c[i, j]))
                assert abs(rho_d[i, j]) <= abs(rho_c[i, j]) + 1e-12


# ---- pipeline convenience (increment 1 + 2) --------------------------------

def test_build_exponential_target_matches_manual_chain():
    rho_c = _hermitian_rho(4, seed=9)
    favg = np.array([[0.0], [1.0], [2.5], [-1.0]])
    alpha, dt, hbar = 1.3, 0.6, 1.0
    tau = decoherence_time_matrix(favg, alpha, hbar=hbar)
    expected = exponential_target(rho_c, tau, dt)
    got = build_exponential_target(rho_c, favg, alpha, dt, hbar=hbar)
    np.testing.assert_allclose(got, expected)


def test_two_state_single_step_exact_exponential():
    # 2-state model seed: over one step the coherence scales by exactly
    # exp(-dt/tau) with tau from the constant force difference.
    dF, alpha, dt = 2.0, 1.0, 0.25
    favg = np.array([[0.0], [dF]])
    tau = np.sqrt(8.0 * alpha) / dF  # hbar = 1
    rho_c = np.array([[0.6, 0.3 + 0.2j],
                      [0.3 - 0.2j, 0.4]])
    rho_d = build_exponential_target(rho_c, favg, alpha, dt)
    assert rho_d[0, 1] == pytest.approx(rho_c[0, 1] * np.exp(-dt / tau))


# ---- input validation ------------------------------------------------------

def test_shape_mismatch_raises():
    rho_c = _hermitian_rho(3)
    tau = np.full((2, 2), np.inf)
    with pytest.raises(ValueError):
        exponential_target(rho_c, tau, dt=0.1)


def test_non_square_rho_raises():
    with pytest.raises(ValueError):
        exponential_target(np.zeros((2, 3)), np.zeros((2, 3)), dt=0.1)
