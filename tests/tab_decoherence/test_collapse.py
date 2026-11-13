import numpy as np
import pytest

from tides.decoherence import (
    select_block,
    collapse_wavefunction,
    stochastic_collapse,
    adiabatic_energy,
    greedy_block_decomposition,
    build_exponential_target,
)

# Increment 4 (collapse) of the rTAB-w1 build (see .claude/skills/tab-decoherence).
# Verified against J. Chem. Phys. 152, 234105 (2020), Eqs. 26-27 (selection) and
# Eqs. 10-11 (amplitude collapse).


# ---- block selection by cumulative rule (Eq. 27) ---------------------------

def test_select_block_boundaries():
    w = np.array([0.3, 0.5, 0.2])
    # cumulative boundaries: [0.3, 0.8, 1.0]
    assert select_block(w, gamma=0.0) == 0
    assert select_block(w, gamma=0.29) == 0
    assert select_block(w, gamma=0.3) == 1     # 0.3 <= gamma opens block 1
    assert select_block(w, gamma=0.79) == 1
    assert select_block(w, gamma=0.8) == 2
    assert select_block(w, gamma=0.999) == 2


def test_select_block_skips_zero_weight():
    w = np.array([0.4, 0.0, 0.6])
    # block 1 has zero weight and must never be chosen
    assert select_block(w, gamma=0.4) == 2
    assert select_block(w, gamma=0.399) == 0
    for g in np.linspace(0, 0.9999, 50):
        assert select_block(w, gamma=g) != 1


def test_select_block_frequencies_match_weights():
    w = np.array([0.1, 0.6, 0.25, 0.05])
    rng = np.random.default_rng(0)
    draws = np.array([select_block(w, rng=rng) for _ in range(20000)])
    freq = np.bincount(draws, minlength=4) / draws.size
    np.testing.assert_allclose(freq, w, atol=0.02)


def test_select_block_rejects_bad_gamma():
    with pytest.raises(ValueError):
        select_block([0.5, 0.5], gamma=1.0)
    with pytest.raises(ValueError):
        select_block([0.5, 0.5], gamma=-0.1)


# ---- amplitude collapse (Eqs. 10-11) ---------------------------------------

def test_collapse_projects_and_renormalizes():
    c = np.array([0.5, 0.5, 0.5, 0.5], dtype=complex)  # normalized
    # a block supported on states {0, 2}
    block = np.zeros((4, 4), dtype=complex)
    block[np.ix_([0, 2], [0, 2])] = 0.5
    c_new = collapse_wavefunction(c, block)
    assert np.linalg.norm(c_new) == pytest.approx(1.0)
    assert c_new[1] == 0 and c_new[3] == 0
    np.testing.assert_allclose(np.abs(c_new[[0, 2]]), [1 / np.sqrt(2)] * 2)


def test_collapse_preserves_relative_phase():
    c = np.array([1.0, 1.0j, 2.0], dtype=complex)
    c = c / np.linalg.norm(c)
    block = np.zeros((3, 3), dtype=complex)
    block[np.ix_([0, 1], [0, 1])] = 1.0  # support {0,1}; values irrelevant to support
    c_new = collapse_wavefunction(c, block)
    # phase ratio between surviving components is unchanged
    assert np.angle(c_new[1] / c_new[0]) == pytest.approx(np.angle(c[1] / c[0]))


def test_collapse_reconstructs_block_for_pure_state():
    # For a pure trajectory, |c_new><c_new| must equal the (pure) block exactly.
    rng = np.random.default_rng(5)
    c = rng.normal(size=4) + 1j * rng.normal(size=4)
    c = c / np.linalg.norm(c)
    rho_c = np.outer(c, c.conj())
    rho_d = build_exponential_target(rho_c, np.arange(4.0)[:, None], alpha=1.0, dt=0.8)
    weights, blocks = greedy_block_decomposition(rho_d, rho_c, rng=np.random.default_rng(1))
    for block in blocks:
        c_new = collapse_wavefunction(c, block)
        np.testing.assert_allclose(np.outer(c_new, c_new.conj()), block, atol=1e-10)


def test_collapse_empty_support_raises():
    c = np.array([1.0, 0.0], dtype=complex)
    block = np.zeros((2, 2))
    with pytest.raises(ValueError):
        collapse_wavefunction(c, block)


# ---- ensemble average reproduces rho^d (the point of TAB) ------------------

def test_ensemble_average_reproduces_target():
    rng = np.random.default_rng(7)
    c = rng.normal(size=3) + 1j * rng.normal(size=3)
    c = c / np.linalg.norm(c)
    rho_c = np.outer(c, c.conj())
    rho_d = build_exponential_target(rho_c, np.array([[0.0], [1.5], [3.0]]),
                                     alpha=1.0, dt=0.6)
    weights, blocks = greedy_block_decomposition(rho_d, rho_c, rng=rng)
    # average |c_new><c_new| over many collapses should approach rho^d
    acc = np.zeros((3, 3), dtype=complex)
    n = 40000
    for _ in range(n):
        _, c_new = stochastic_collapse(c, weights, blocks, rng=rng)
        acc += np.outer(c_new, c_new.conj())
    np.testing.assert_allclose(acc / n, rho_d, atol=0.01)


# ---- adiabatic energy helper -----------------------------------------------

def test_adiabatic_energy_is_population_weighted_sum():
    rho = np.array([[0.25, 0.1j, 0.0],
                    [-0.1j, 0.5, 0.2],
                    [0.0, 0.2, 0.25]], dtype=complex)
    energies = np.array([-1.0, 0.0, 2.0])
    # only diagonal populations contribute
    assert adiabatic_energy(rho, energies) == pytest.approx(0.25 * -1.0 + 0.25 * 2.0)
