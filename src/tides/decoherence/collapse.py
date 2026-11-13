import numpy as np

'''
Stochastic wavefunction collapse into a block.

Increment 4 (part 1) of the rTAB-w1 implementation (see .claude/skills/tab-decoherence).
Given the block weights and block density matrices from the greedy decomposition
(increment 3), one block is chosen at random -- in proportion to its weight -- and the
electronic wavefunction collapses onto it. Over an ensemble of trajectories this
reproduces the target density matrix rho^d, while each individual trajectory stays in
a pure state.

Equations, from the typeset PDFs -- Esch & Levine, J. Chem. Phys. 152, 234105 (2020),
Eqs. 26-27 (equivalently J. Chem. Phys. 153, 114104 (2020), Eqs. 17-18):

    sum_a P^block_a = 1
    draw gamma ~ U[0, 1); collapse into block a for which
        sum_{b<a} P^block_b <= gamma < sum_{b<=a} P^block_b

On collapse the electronic amplitudes of states outside the block's support are
zeroed and the remainder are renormalized, which preserves the relative populations
and phases within the block (A Eqs. 10-11 / B collapse). The energy-conserving nuclear
momentum rescale that must accompany the collapse lives in `rescale.py`.
'''


def select_block(weights, rng=None, gamma=None):
    '''Pick a block index a by the cumulative-weight rule (Eq. 27).

    Parameters
    ----------
    weights : array_like, shape (nblocks,)
        Block weights P^block_a; non-negative and summing to (approximately) 1.
    rng : numpy.random.Generator, optional
        Source of randomness. Defaults to a fresh default_rng. Ignored if `gamma`
        is given.
    gamma : float, optional
        A specific draw in [0, 1) to use instead of sampling (for deterministic
        testing).

    Returns
    -------
    int
        The selected block index. Zero-weight blocks are never selected.
    '''
    weights = np.asarray(weights, dtype=float)
    if weights.ndim != 1:
        raise ValueError('weights must be 1-D')
    if np.any(weights < -1e-9):
        raise ValueError('weights must be non-negative')
    if gamma is None:
        rng = np.random.default_rng() if rng is None else rng
        gamma = rng.random()
    elif not (0.0 <= gamma < 1.0):
        raise ValueError('gamma must lie in [0, 1)')

    cumulative = np.cumsum(weights)
    a = int(np.searchsorted(cumulative, gamma, side='right'))
    # Guard against gamma landing past the last boundary when the weights sum to
    # slightly under 1 due to floating-point error.
    return min(a, len(weights) - 1)


def collapse_wavefunction(c, block, population_threshold=1e-12):
    '''Collapse amplitude vector c onto the support of `block` (A Eqs. 10-11).

    The amplitudes of states outside the block are set to zero and the vector is
    renormalized, preserving the relative magnitudes and phases within the block.
    For a pure Ehrenfest trajectory (block derived from rho^c = |c><c|) the result
    satisfies |c_new><c_new| == block exactly.

    Parameters
    ----------
    c : array_like, shape (nstates,)
        Electronic amplitude vector before collapse.
    block : array_like, shape (nstates, nstates)
        The block density matrix to collapse into; its non-zero diagonal entries
        define the block's state support.
    population_threshold : float, optional
        Diagonal magnitude above which a state counts as part of the block support.

    Returns
    -------
    ndarray, shape (nstates,)
        The renormalized post-collapse amplitude vector.
    '''
    c = np.asarray(c, dtype=complex)
    block = np.asarray(block)
    support = np.abs(np.diagonal(block)) > population_threshold
    c_new = np.where(support, c, 0.0)
    norm = np.linalg.norm(c_new)
    if norm == 0.0:
        raise ValueError('collapse target has empty support or zero amplitude on it')
    return c_new / norm


def stochastic_collapse(c, weights, blocks, rng=None, gamma=None):
    '''Select a block and collapse c onto it (Eqs. 26-27 then Eqs. 10-11).

    Convenience wrapper chaining `select_block` and `collapse_wavefunction`.

    Returns
    -------
    a : int
        Index of the selected block.
    c_new : ndarray, shape (nstates,)
        Post-collapse amplitude vector.
    '''
    a = select_block(weights, rng=rng, gamma=gamma)
    c_new = collapse_wavefunction(c, blocks[a])
    return a, c_new


def adiabatic_energy(rho, energies):
    '''Electronic energy Tr(rho H) in the adiabatic basis (H diagonal).

    In the adiabatic basis the electronic Hamiltonian is diagonal, so the energy is
    sum_i rho_ii E_i and the coherences do not contribute. This is the quantity whose
    change across a collapse the momentum rescale must compensate.

    Parameters
    ----------
    rho : array_like, shape (nstates, nstates)
        Electronic density matrix.
    energies : array_like, shape (nstates,)
        Adiabatic state energies E_i.

    Returns
    -------
    float
        The electronic energy.
    '''
    rho = np.asarray(rho)
    energies = np.asarray(energies, dtype=float)
    return float(np.sum(np.diagonal(rho).real * energies))
