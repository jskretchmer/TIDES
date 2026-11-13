import numpy as np
import pytest
from scipy.special import erfc

from tides.decoherence import (
    tabw1_decay_factor,
    tabw1_target,
    build_tabw1_target,
    exponential_target,
    decoherence_time_matrix,
    greedy_block_decomposition,
)

# Increment 5 of the rTAB-w1 build (see .claude/skills/tab-decoherence).
# TAB-w1 target, verified against J. Chem. Phys. 155, 214101 (2021), Eqs. 7-12.


def _pure_density(amps):
    c = np.asarray(amps, dtype=complex)
    c = c / np.linalg.norm(c)
    return np.outer(c, c.conj())


def _eq9_decay_numerical(tau, dt, pk, pdotk, ngrid=200000):
    '''Directly integrate Eq. 9 with the w1 step function, for one pair.

    Independent reference implementation used to validate the erf closed form.
    Requires pdotk > 0 (non-empty w1 history).
    '''
    t0 = -pk / pdotk            # earliest formation time in the history window
    tprime = np.linspace(t0, 0.0, ngrid)
    t = np.linspace(0.0, dt, ngrid)
    # w1 is the constant rhodot/rho on the window; it cancels, but keep it explicit.
    w = pdotk / pk

    def G(tt, tp):
        return np.exp(-(tt - tp) ** 2 / tau ** 2)

    # inner integral over t of Gdot = G(dt, t') - G(0, t')  (fundamental theorem)
    inner = G(dt, tprime) - G(0.0, tprime)
    numerator = np.trapz(w * inner, tprime)
    denominator = np.trapz(w * G(0.0, tprime), tprime)
    return 1.0 + numerator / denominator


# ---- closed form matches direct integration of Eq. 9 -----------------------

@pytest.mark.parametrize('tau,dt,pk,pdotk', [
    (1.0, 0.1, 0.5, 0.4),
    (2.0, 0.5, 0.8, 0.05),
    (0.7, 0.3, 0.6, 2.0),
    (5.0, 1.0, 0.3, 0.2),
    (1.5, 0.25, 0.9, 0.01),
])
def test_closed_form_matches_numerical_eq9(tau, dt, pk, pdotk):
    tau_mat = np.array([[np.inf, tau], [tau, np.inf]])
    pops = np.array([pk, 1.0 - pk])
    # make state 0 the faster-changing one so k = 0 with rho^c_00 = pk
    pdots = np.array([pdotk, -pdotk])
    decay = tabw1_decay_factor(tau_mat, dt, pops, pdots)
    expected = _eq9_decay_numerical(tau, dt, pk, pdotk)
    assert decay[0, 1] == pytest.approx(expected, rel=1e-4)
    assert decay[0, 1] == decay[1, 0]  # symmetric


# ---- limiting cases --------------------------------------------------------

def test_fresh_coherence_is_pure_gaussian():
    # T = pk/pdotk -> 0 (huge derivative) -> decay -> exp(-dt^2/tau^2)
    tau, dt = 1.3, 0.4
    tau_mat = np.array([[np.inf, tau], [tau, np.inf]])
    pops = np.array([0.5, 0.5])
    pdots = np.array([1e6, -1e6])  # T ~ 5e-7 -> essentially fresh
    decay = tabw1_decay_factor(tau_mat, dt, pops, pdots)
    assert decay[0, 1] == pytest.approx(np.exp(-dt ** 2 / tau ** 2), rel=1e-5)


def test_old_coherence_tends_to_erfc():
    # T -> inf (pdotk -> 0+) -> decay -> erfc(dt/tau)
    tau, dt = 1.1, 0.5
    tau_mat = np.array([[np.inf, tau], [tau, np.inf]])
    pops = np.array([0.5, 0.5])
    pdots = np.array([1e-8, -1e-8])  # T ~ 5e7
    decay = tabw1_decay_factor(tau_mat, dt, pops, pdots)
    assert decay[0, 1] == pytest.approx(erfc(dt / tau), rel=1e-4)


def test_empty_history_fallback_erfc():
    # rhodot_kk <= 0 for both states -> empty w1 history -> erfc fallback
    tau, dt = 1.0, 0.5
    tau_mat = np.array([[np.inf, tau], [tau, np.inf]])
    pops = np.array([0.5, 0.5])
    pdots = np.array([-0.1, -0.2])
    decay = tabw1_decay_factor(tau_mat, dt, pops, pdots)
    assert decay[0, 1] == pytest.approx(erfc(dt / tau))


def test_empty_history_fallback_one():
    tau, dt = 1.0, 0.5
    tau_mat = np.array([[np.inf, tau], [tau, np.inf]])
    pops = np.array([0.5, 0.5])
    pdots = np.array([-0.1, -0.2])
    decay = tabw1_decay_factor(tau_mat, dt, pops, pdots, no_history_decay='one')
    assert decay[0, 1] == pytest.approx(1.0)


def test_parallel_states_no_decoherence():
    tau_mat = np.full((2, 2), np.inf)
    decay = tabw1_decay_factor(tau_mat, 0.5, np.array([0.5, 0.5]), np.array([0.3, -0.3]))
    np.testing.assert_allclose(decay, 1.0)


# ---- the physical point: Gaussian decays slower than exponential at short dt --

def test_gaussian_slower_than_exponential_short_step():
    tau, dt = 1.0, 0.2  # dt < tau: short-time regime
    tau_mat = np.array([[np.inf, tau], [tau, np.inf]])
    pops = np.array([0.5, 0.5])
    pdots = np.array([1e6, -1e6])  # fresh coherence -> Gaussian
    w1_decay = tabw1_decay_factor(tau_mat, dt, pops, pdots)[0, 1]
    exp_decay = np.exp(-dt / tau)
    # w1 (Gaussian) retains more coherence than the plain exponential
    assert w1_decay > exp_decay


# ---- structural guarantees -------------------------------------------------

def test_decay_in_unit_interval():
    rng = np.random.default_rng(0)
    tau_mat = decoherence_time_matrix(rng.normal(size=(5, 2)), alpha=1.0)
    pops = rng.random(5)
    pops /= pops.sum()
    pdots = rng.normal(size=5)
    decay = tabw1_decay_factor(tau_mat, 0.5, pops, pdots)
    assert np.all(decay > 0.0) and np.all(decay <= 1.0 + 1e-12)


def test_target_preserves_hermiticity_trace_and_diagonal():
    rng = np.random.default_rng(1)
    c = rng.normal(size=4) + 1j * rng.normal(size=4)
    rho_c = _pure_density(c)
    tau = decoherence_time_matrix(rng.normal(size=(4, 2)), alpha=1.0)
    pdots = rng.normal(size=4)
    rho_d = tabw1_target(rho_c, tau, dt=0.6, pop_deriv=pdots)
    np.testing.assert_allclose(rho_d, rho_d.conj().T, atol=1e-12)
    assert np.trace(rho_d) == pytest.approx(np.trace(rho_c))
    np.testing.assert_allclose(np.diagonal(rho_d), np.diagonal(rho_c))


def test_only_magnitude_scaled_phase_preserved():
    rng = np.random.default_rng(2)
    rho_c = _pure_density(rng.normal(size=3) + 1j * rng.normal(size=3))
    tau = np.array([[np.inf, 1.0, 2.0], [1.0, np.inf, 1.5], [2.0, 1.5, np.inf]])
    rho_d = tabw1_target(rho_c, tau, dt=0.4, pop_deriv=np.array([1.0, -0.5, -0.5]))
    for i in range(3):
        for j in range(3):
            if i != j and abs(rho_c[i, j]) > 0:
                assert np.angle(rho_d[i, j]) == pytest.approx(np.angle(rho_c[i, j]))
                assert abs(rho_d[i, j]) <= abs(rho_c[i, j]) + 1e-12


# ---- why the greedy decomposition (increment 3) is NOT used for w1 ----------
#
# Unlike the plain-exponential target (which stays positive-semidefinite and is
# always greedy-decomposable), the TAB-w1 target can be non-PSD because different
# pairs decay by inconsistent factors. A non-PSD target cannot be written as a
# non-negative combination of PSD blocks, so the greedy (aTAB) decomposition gives
# negative / non-convergent weights on it. This is exactly why Paper C introduces
# the non-negative least-squares rTAB algorithm (increment 6). The w1 -> collapse
# reconstruction test therefore lives with rTAB, not here.

def test_w1_target_can_be_non_psd_motivating_rtab():
    # A concrete non-PSD w1 target (contrast with the always-PSD exponential one).
    rng = np.random.default_rng(0)
    rho_c = _pure_density(rng.normal(size=5) + 1j * rng.normal(size=5))
    tau = decoherence_time_matrix(rng.normal(size=(5, 1)), alpha=1.0)
    pdot = rng.normal(size=5)
    rho_w1 = tabw1_target(rho_c, tau, dt=0.5, pop_deriv=pdot)
    rho_exp = exponential_target(rho_c, tau, dt=0.5)
    # w1 target dips below PSD; exponential target does not
    assert np.linalg.eigvalsh(rho_w1).min() < 0.0
    assert np.linalg.eigvalsh(rho_exp).min() >= -1e-12


def test_greedy_not_robust_on_w1_target():
    # Documents that the greedy decomposition can fail to converge on w1 targets,
    # justifying rTAB as the required decomposer for the rTAB-w1 method.
    rng = np.random.default_rng(3)
    rho_c = _pure_density(rng.normal(size=5) + 1j * rng.normal(size=5))
    tau = decoherence_time_matrix(rng.normal(size=(5, 2)), alpha=1.0)
    rho_d = tabw1_target(rho_c, tau, dt=0.5, pop_deriv=rng.normal(size=5))
    with pytest.raises(RuntimeError):
        greedy_block_decomposition(rho_d, rho_c, rng=rng)


# ---- convenience + validation ----------------------------------------------

def test_build_tabw1_matches_manual_chain():
    rng = np.random.default_rng(4)
    rho_c = _pure_density(rng.normal(size=4) + 1j * rng.normal(size=4))
    favg = rng.normal(size=(4, 2))
    pdots = rng.normal(size=4)
    tau = decoherence_time_matrix(favg, alpha=1.2, hbar=1.0)
    expected = tabw1_target(rho_c, tau, dt=0.7, pop_deriv=pdots)
    got = build_tabw1_target(rho_c, favg, alpha=1.2, dt=0.7, pop_deriv=pdots)
    np.testing.assert_allclose(got, expected)


def test_invalid_fallback_raises():
    tau = np.array([[np.inf, 1.0], [1.0, np.inf]])
    with pytest.raises(ValueError):
        tabw1_decay_factor(tau, 0.5, np.array([0.5, 0.5]), np.array([1.0, -1.0]),
                           no_history_decay='bogus')


def test_shape_mismatch_raises():
    rho_c = _pure_density([1.0, 1.0, 1.0])
    with pytest.raises(ValueError):
        tabw1_target(rho_c, np.full((2, 2), np.inf), dt=0.1, pop_deriv=np.zeros(2))
