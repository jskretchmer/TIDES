import numpy as np
import pytest

from tides.decoherence import (
    greedy_block_decomposition,
    exponential_target,
    decoherence_time_matrix,
    build_exponential_target,
)

# Increment 3 of the rTAB-w1 build (see .claude/skills/tab-decoherence).
# Greedy block decomposition, verified against the typeset PDFs:
#   J. Chem. Phys. 153, 114104 (2020), Eqs. 11-16 (primary)
#   J. Chem. Phys. 152, 234105 (2020), Eqs. 19-25 (cross-check)
#
# The central correctness property is reconstruction:
#     sum_a P^block_a rho^block_a == rho^d.


def _pure_density(amps):
    '''Pure-state density matrix |psi><psi| from an amplitude vector.'''
    c = np.asarray(amps, dtype=complex)
    c = c / np.linalg.norm(c)
    return np.outer(c, c.conj())


def _random_density(nstates, seed):
    '''A random valid (Hermitian, PSD, unit-trace) density matrix.'''
    rng = np.random.default_rng(seed)
    a = rng.normal(size=(nstates, nstates)) + 1j * rng.normal(size=(nstates, nstates))
    rho = a @ a.conj().T
    return rho / np.trace(rho).real


def _reconstruct(weights, blocks):
    if len(weights) == 0:
        return np.zeros((0, 0))
    return np.tensordot(weights, blocks, axes=(0, 0))


# ---- reconstruction: the core invariant ------------------------------------

@pytest.mark.parametrize('nstates', [2, 3, 4, 5])
@pytest.mark.parametrize('dt', [0.0, 0.1, 0.5, 2.0, 20.0])
def test_reconstruction_random_density(nstates, dt):
    rng = np.random.default_rng(100 + nstates)
    rho_c = _random_density(nstates, seed=7 * nstates)
    favg = rng.normal(size=(nstates, 2))
    tau = decoherence_time_matrix(favg, alpha=1.0)
    rho_d = exponential_target(rho_c, tau, dt)
    w, blocks = greedy_block_decomposition(rho_d, rho_c, rng=rng)
    np.testing.assert_allclose(_reconstruct(w, blocks), rho_d, atol=1e-10)


@pytest.mark.parametrize('nstates', [2, 3, 5])
def test_reconstruction_pure_state(nstates):
    rng = np.random.default_rng(3)
    amps = rng.normal(size=nstates) + 1j * rng.normal(size=nstates)
    rho_c = _pure_density(amps)
    favg = np.arange(nstates, dtype=float)[:, None]
    rho_d = build_exponential_target(rho_c, favg, alpha=1.0, dt=0.7)
    w, blocks = greedy_block_decomposition(rho_d, rho_c, rng=rng)
    np.testing.assert_allclose(_reconstruct(w, blocks), rho_d, atol=1e-10)


# ---- weight and block properties -------------------------------------------

def test_weights_sum_to_one():
    rng = np.random.default_rng(11)
    rho_c = _random_density(5, seed=21)
    rho_d = build_exponential_target(rho_c, rng.normal(size=(5, 3)), alpha=1.0, dt=0.4)
    w, _ = greedy_block_decomposition(rho_d, rho_c, rng=rng)
    assert w.sum() == pytest.approx(1.0)


def test_weights_non_negative():
    rng = np.random.default_rng(12)
    rho_c = _random_density(5, seed=22)
    rho_d = build_exponential_target(rho_c, rng.normal(size=(5, 3)), alpha=1.0, dt=0.9)
    w, _ = greedy_block_decomposition(rho_d, rho_c, rng=rng)
    assert np.all(w >= -1e-12)


def test_each_block_has_unit_trace():
    rng = np.random.default_rng(13)
    rho_c = _random_density(4, seed=23)
    rho_d = build_exponential_target(rho_c, rng.normal(size=(4, 2)), alpha=1.0, dt=0.6)
    _, blocks = greedy_block_decomposition(rho_d, rho_c, rng=rng)
    for block in blocks:
        assert np.trace(block) == pytest.approx(1.0)


def test_each_block_hermitian_and_psd():
    rng = np.random.default_rng(14)
    rho_c = _random_density(5, seed=24)
    rho_d = build_exponential_target(rho_c, rng.normal(size=(5, 2)), alpha=1.0, dt=0.5)
    _, blocks = greedy_block_decomposition(rho_d, rho_c, rng=rng)
    for block in blocks:
        np.testing.assert_allclose(block, block.conj().T, atol=1e-12)
        eigvals = np.linalg.eigvalsh(block)
        assert eigvals.min() > -1e-10


# ---- limiting cases with hand-checkable answers ----------------------------

def test_no_decoherence_gives_single_full_block():
    # dt = 0 -> rho^d = rho^c -> one block equal to rho^c with weight 1.
    rho_c = _pure_density([1.0, 1.0, 1.0])
    rho_d = exponential_target(rho_c, np.full((3, 3), np.inf), dt=1.0)
    w, blocks = greedy_block_decomposition(rho_d, rho_c,
                                           rng=np.random.default_rng(0))
    assert len(w) == 1
    assert w[0] == pytest.approx(1.0)
    np.testing.assert_allclose(blocks[0], rho_c, atol=1e-12)


def test_full_decoherence_gives_single_state_blocks():
    # tau -> 0 kills all coherences -> rho^d diagonal -> single-state blocks whose
    # weights are the populations.
    rho_c = _pure_density([1.0, 1.0j, 1.0])       # populations 1/3 each
    tau = np.full((3, 3), np.inf)
    tau[~np.eye(3, dtype=bool)] = 1e-14           # off-diagonals decohere instantly
    rho_d = exponential_target(rho_c, tau, dt=1.0)
    w, blocks = greedy_block_decomposition(rho_d, rho_c,
                                           rng=np.random.default_rng(0))
    # every block supports exactly one state
    for block in blocks:
        support = np.where(np.abs(np.diagonal(block)) > 1e-9)[0]
        assert len(support) == 1
    np.testing.assert_allclose(_reconstruct(w, blocks), rho_d, atol=1e-12)
    np.testing.assert_allclose(np.sort(w), np.full(3, 1 / 3), atol=1e-9)


def test_two_state_symmetric_weights_hand_value():
    # Symmetric pure superposition; decay factor d on the coherence.
    # Expected decomposition: full-coherent block (weight d) + two single-state
    # blocks (weight (1-d)/2 each).
    rho_c = _pure_density([1.0, 1.0])
    dt, tau = 0.5, 1.3
    d = np.exp(-dt / tau)
    tau_mat = np.array([[np.inf, tau], [tau, np.inf]])
    rho_d = exponential_target(rho_c, tau_mat, dt)
    w, blocks = greedy_block_decomposition(rho_d, rho_c,
                                           rng=np.random.default_rng(0))
    np.testing.assert_allclose(_reconstruct(w, blocks), rho_d, atol=1e-12)
    assert np.sort(w)[-1] == pytest.approx(max(d, (1 - d) / 2))
    # the fully coherent block (support on both states) carries weight d
    coh = [w[a] for a, blk in enumerate(blocks)
           if abs(blk[0, 1]) > 1e-9]
    assert sum(coh) == pytest.approx(d)


def test_parallel_states_share_a_coherent_block():
    # States 0,1 parallel (never decohere from each other); state 2 decoheres from
    # both. Some block must retain the 0-1 coherence.
    rho_c = _pure_density([1.0, 1.0, 1.0])
    tau = np.array([[np.inf, np.inf, 1.0],
                    [np.inf, np.inf, 1.0],
                    [1.0, 1.0, np.inf]])
    rho_d = exponential_target(rho_c, tau, dt=1.0)
    w, blocks = greedy_block_decomposition(rho_d, rho_c,
                                           rng=np.random.default_rng(0))
    retains_01 = [a for a, blk in enumerate(blocks) if abs(blk[0, 1]) > 1e-9]
    assert retains_01, 'no block retained the parallel-state coherence'
    # reconstruction still exact
    np.testing.assert_allclose(_reconstruct(w, blocks), rho_d, atol=1e-12)


# ---- reproducibility and validation ----------------------------------------

def test_deterministic_with_seeded_rng():
    rho_c = _random_density(6, seed=99)
    rho_d = build_exponential_target(
        rho_c, np.random.default_rng(1).normal(size=(6, 3)), alpha=1.0, dt=0.5)
    w1, b1 = greedy_block_decomposition(rho_d, rho_c, rng=np.random.default_rng(42))
    w2, b2 = greedy_block_decomposition(rho_d, rho_c, rng=np.random.default_rng(42))
    np.testing.assert_allclose(w1, w2)
    np.testing.assert_allclose(b1, b2)


def test_shape_mismatch_raises():
    with pytest.raises(ValueError):
        greedy_block_decomposition(np.zeros((3, 3)), np.zeros((2, 2)))


def test_non_square_raises():
    with pytest.raises(ValueError):
        greedy_block_decomposition(np.zeros((2, 3)), np.zeros((2, 3)))
