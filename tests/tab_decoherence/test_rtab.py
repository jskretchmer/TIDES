import numpy as np
import pytest

from tides.decoherence import (
    rtab_decomposition,
    exponential_target,
    tabw1_target,
    build_tabw1_target,
    decoherence_time_matrix,
    stochastic_collapse,
)

# Increment 6 of the rTAB-w1 build (see .claude/skills/tab-decoherence).
# rTAB non-negative least-squares decomposition, verified against
# J. Chem. Phys. 155, 214101 (2021), Sec. II D, Eq. 17.


def _pure_density(amps):
    c = np.asarray(amps, dtype=complex)
    c = c / np.linalg.norm(c)
    return np.outer(c, c.conj())


def _reconstruct(weights, blocks):
    return np.tensordot(weights, blocks, axes=(0, 0))


# ---- exact reconstruction of the (PSD) exponential target ------------------

@pytest.mark.parametrize('nstates', [2, 3, 4, 5])
def test_rtab_reconstructs_exponential_target(nstates):
    rng = np.random.default_rng(nstates)
    rho_c = _pure_density(rng.normal(size=nstates) + 1j * rng.normal(size=nstates))
    tau = decoherence_time_matrix(rng.normal(size=(nstates, 2)), alpha=1.0)
    rho_d = exponential_target(rho_c, tau, dt=0.5)
    w, blocks = rtab_decomposition(rho_d, rho_c, rng=rng)
    # exponential target is exactly non-negative-decomposable -> NNLS is exact
    np.testing.assert_allclose(_reconstruct(w, blocks), rho_d, atol=1e-6)


# ---- robustness on the (possibly non-PSD) w1 target ------------------------

def test_rtab_never_fails_on_w1_targets():
    # The whole reason rTAB exists: greedy raises on these, rTAB must not.
    for s in range(40):
        rng = np.random.default_rng(s)
        rho_c = _pure_density(rng.normal(size=5) + 1j * rng.normal(size=5))
        tau = decoherence_time_matrix(rng.normal(size=(5, 2)), alpha=1.0)
        rho_d = tabw1_target(rho_c, tau, dt=0.5, pop_deriv=rng.normal(size=5))
        w, blocks = rtab_decomposition(rho_d, rho_c, rng=rng)
        assert np.all(w >= 0.0)
        assert w.sum() == pytest.approx(1.0)


def test_rtab_conserves_populations_on_w1_target():
    # Diagonal scaling (1e4) enforces population conservation even when the full
    # target is only approximately representable.
    rng = np.random.default_rng(0)
    rho_c = _pure_density(rng.normal(size=5) + 1j * rng.normal(size=5))
    tau = decoherence_time_matrix(rng.normal(size=(5, 2)), alpha=1.0)
    rho_d = tabw1_target(rho_c, tau, dt=0.5, pop_deriv=rng.normal(size=5))
    w, blocks = rtab_decomposition(rho_d, rho_c, rng=rng)
    recon = _reconstruct(w, blocks)
    np.testing.assert_allclose(np.diagonal(recon).real, np.diagonal(rho_d).real, atol=1e-4)


# ---- weight and block properties -------------------------------------------

def test_rtab_weights_non_negative_and_normalized():
    rng = np.random.default_rng(2)
    rho_c = _pure_density(rng.normal(size=5) + 1j * rng.normal(size=5))
    rho_d = build_tabw1_target(rho_c, rng.normal(size=(5, 2)), alpha=1.0, dt=0.7,
                               pop_deriv=rng.normal(size=5))
    w, blocks = rtab_decomposition(rho_d, rho_c, rng=rng)
    assert np.all(w >= 0.0)
    assert w.sum() == pytest.approx(1.0)


def test_rtab_blocks_are_valid_density_matrices():
    rng = np.random.default_rng(3)
    rho_c = _pure_density(rng.normal(size=4) + 1j * rng.normal(size=4))
    rho_d = build_tabw1_target(rho_c, rng.normal(size=(4, 2)), alpha=1.0, dt=0.5,
                               pop_deriv=rng.normal(size=4))
    _, blocks = rtab_decomposition(rho_d, rho_c, rng=rng)
    for block in blocks:
        assert np.trace(block) == pytest.approx(1.0)
        np.testing.assert_allclose(block, block.conj().T, atol=1e-12)
        assert np.linalg.eigvalsh(block).min() > -1e-10


# ---- end-to-end rTAB-w1: ensemble average reproduces populations -----------

def test_rtab_w1_ensemble_reproduces_populations():
    # The full rTAB-w1 pipeline: tau -> w1 target -> rTAB -> stochastic collapse.
    # Populations are conserved exactly in expectation; this is the end-to-end
    # correctness check greedy could not provide for w1.
    rng = np.random.default_rng(11)
    c = rng.normal(size=3) + 1j * rng.normal(size=3)
    rho_c = _pure_density(c)
    rho_d = build_tabw1_target(rho_c, np.array([[0.0], [1.5], [3.0]]), alpha=1.0,
                               dt=0.5, pop_deriv=np.array([0.4, -0.1, -0.3]))
    weights, blocks = rtab_decomposition(rho_d, rho_c, rng=rng)
    acc = np.zeros(3)
    n = 40000
    for _ in range(n):
        _, c_new = stochastic_collapse(c, weights, blocks, rng=rng)
        acc += np.abs(c_new) ** 2
    np.testing.assert_allclose(acc / n, np.diagonal(rho_d).real, atol=0.02)


# ---- agreement with greedy on the exponential target -----------------------

def test_rtab_matches_greedy_reconstruction_on_exponential():
    rng = np.random.default_rng(9)
    rho_c = _pure_density(rng.normal(size=4) + 1j * rng.normal(size=4))
    tau = decoherence_time_matrix(rng.normal(size=(4, 1)), alpha=1.0)
    rho_d = exponential_target(rho_c, tau, dt=0.6)
    w, blocks = rtab_decomposition(rho_d, rho_c, rng=rng)
    # both should reconstruct rho^d; rTAB via NNLS, greedy exactly
    np.testing.assert_allclose(_reconstruct(w, blocks), rho_d, atol=1e-6)


# ---- validation ------------------------------------------------------------

def test_rtab_shape_mismatch_raises():
    with pytest.raises(ValueError):
        rtab_decomposition(np.zeros((3, 3)), np.zeros((2, 2)))


def test_rtab_deterministic_with_seeded_rng():
    rng_target = np.random.default_rng(5)
    rho_c = _pure_density(rng_target.normal(size=5) + 1j * rng_target.normal(size=5))
    rho_d = build_tabw1_target(rho_c, rng_target.normal(size=(5, 2)), alpha=1.0,
                               dt=0.5, pop_deriv=rng_target.normal(size=5))
    w1, b1 = rtab_decomposition(rho_d, rho_c, rng=np.random.default_rng(7))
    w2, b2 = rtab_decomposition(rho_d, rho_c, rng=np.random.default_rng(7))
    np.testing.assert_allclose(w1, w2)
    np.testing.assert_allclose(b1, b2)
