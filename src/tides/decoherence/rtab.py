import numpy as np
from scipy.optimize import nnls

from tides.decoherence.blocks import _seed_index, _prune_incoherent_pairs

'''
rTAB: constrained-least-squares collapse probabilities.

Increment 6 of the rTAB-w1 implementation (see .claude/skills/tab-decoherence). This
replaces the greedy weighting of increment 3 with a non-negative least-squares fit,
which is required for the TAB-w1 target: that target can be non-PSD, so the greedy
(aTAB) weights go negative or fail to converge on it. rTAB is Paper C's default and
the decomposer for the rTAB-w1 method.

Esch & Levine, J. Chem. Phys. 155, 214101 (2021), Sec. II D, Eq. 17:

  The diagonal and lower-triangular elements of rho^d are unfolded into a column
  vector r, with the diagonal elements scaled by a large factor (10000) so that the
  expectation value of each state's population is conserved. A matrix A is built whose
  columns are the same unfolding of each candidate rho^block. The collapse
  probabilities P solve the non-negative least-squares problem

      A P = r,   subject to  P >= 0.                                    (Eq. 17)

  Only states with population above 1e-6 are included, and only weights above 1e-6 are
  kept. rTAB restricts the candidate blocks to the "restricted basis of blocks defined
  by the original greedy algorithm" (as opposed to cTAB's full 2^R-1 power set).

Because Hermitian rho^d and Hermitian blocks are fully determined by their diagonal
and lower triangle, fitting the real diagonal plus the real and imaginary parts of the
lower triangle captures the whole matrix. The candidate basis always includes the
single-state blocks, guaranteeing a feasible (fully-decohered) solution exists.
'''


def _generate_candidate_blocks(rho_d, rho_c, threshold, coherence_threshold,
                               neg_weight_tol, population_threshold, rng):
    '''Restricted block basis: greedy-generated blocks plus single-state blocks.

    Runs the greedy seed/grow/prune enumeration (increment 3), but marks each seed
    position resolved so it is never revisited -- guaranteeing termination in at most
    N(N+1)/2 iterations even for non-PSD targets where the greedy *weights* would not
    converge. Blocks are deduplicated by their state support. Single-state blocks for
    every populated state are always included so the least-squares fit is feasible.
    '''
    n = rho_c.shape[0]
    rho_c = rho_c.astype(complex)
    rho_tmp = np.array(rho_d, dtype=complex)
    pops = np.diagonal(rho_c).real
    exists = np.abs(rho_c) > coherence_threshold
    resolved = np.zeros((n, n), dtype=bool)

    blocks_by_support = {}
    # single-state blocks for populated states (guarantee a feasible solution)
    for i in range(n):
        if pops[i] > population_threshold:
            single = np.zeros((n, n), dtype=complex)
            single[i, i] = 1.0
            blocks_by_support[frozenset([i])] = single

    for _ in range(n * (n + 1) // 2 + 1):
        b = np.zeros((n, n))
        b[exists] = (rho_tmp[exists] / rho_c[exists]).real
        b[resolved] = 0.0
        seed = _seed_index(b, threshold)
        if seed is None:
            break
        k, l = seed

        beta = {k, l}
        for p in range(n):
            if p != k and p != l and abs(b[p, l]) > threshold and abs(b[p, k]) > threshold:
                beta.add(p)
        beta = _prune_incoherent_pairs(beta, k, l, b, threshold, rng)
        beta_list = sorted(beta)

        pop_sum = pops[beta_list].sum()
        block = np.zeros((n, n), dtype=complex)
        sub = np.ix_(beta_list, beta_list)
        block[sub] = rho_c[sub] / pop_sum
        blocks_by_support[frozenset(beta)] = block

        P = (rho_tmp[k, l] / block[k, l]).real
        if -neg_weight_tol < P < 0.0:
            P = 0.0
        rho_tmp = rho_tmp - P * block
        resolved[k, l] = resolved[l, k] = True  # never revisit this seed position

    return list(blocks_by_support.values())


def _unfold(mat, diagonal_weight):
    '''Unfold a Hermitian matrix into the real vector r / column of A (Eq. 17).

    Layout: [diagonal_weight * diag] ++ [Re(lower triangle)] ++ [Im(lower triangle)].
    '''
    mat = np.asarray(mat)
    n = mat.shape[0]
    diag = diagonal_weight * np.diagonal(mat).real
    i, j = np.tril_indices(n, k=-1)
    lower = mat[i, j]
    return np.concatenate([diag, lower.real, lower.imag])


def rtab_decomposition(rho_d, rho_c, diagonal_weight=1e4, population_threshold=1e-6,
                       weight_threshold=1e-6, threshold=1e-9, neg_weight_tol=1e-7,
                       coherence_threshold=1e-12, rng=None):
    '''rTAB non-negative least-squares block decomposition (Eq. 17).

    Drop-in replacement for `greedy_block_decomposition`, robust to non-PSD targets
    (i.e. the TAB-w1 target). Returns collapse probabilities and block density
    matrices in the same (weights, blocks) form, so it feeds `stochastic_collapse`
    unchanged.

    Parameters
    ----------
    rho_d : array_like, shape (nstates, nstates)
        Target density matrix (e.g. from `tabw1_target`).
    rho_c : array_like, shape (nstates, nstates)
        Fully coherent Ehrenfest density matrix at the end of the step.
    diagonal_weight : float, optional
        Factor applied to the diagonal (population) equations so populations are
        conserved in expectation (default 1e4, the Paper C value).
    population_threshold : float, optional
        States with rho^c population at or below this are excluded (default 1e-6).
    weight_threshold : float, optional
        Collapse probabilities at or below this are dropped before renormalizing
        (default 1e-6).
    threshold, neg_weight_tol, coherence_threshold : float, optional
        Passed through to the greedy block-basis generation.
    rng : numpy.random.Generator, optional
        Randomness for the block-basis pruning. Defaults to a fresh default_rng.

    Returns
    -------
    weights : ndarray, shape (nblocks,)
        Non-negative collapse probabilities, summing to 1.
    blocks : ndarray, shape (nblocks, nstates, nstates)
        The corresponding block density matrices (Hermitian, unit trace).
    '''
    rho_d = np.asarray(rho_d)
    rho_c = np.asarray(rho_c)
    if rho_d.ndim != 2 or rho_d.shape[0] != rho_d.shape[1]:
        raise ValueError(f'rho_d must be a square matrix; got shape {rho_d.shape}')
    if rho_c.shape != rho_d.shape:
        raise ValueError(f'rho_c shape {rho_c.shape} does not match rho_d shape {rho_d.shape}')
    if rng is None:
        rng = np.random.default_rng()

    candidates = _generate_candidate_blocks(
        rho_d, rho_c, threshold, coherence_threshold, neg_weight_tol,
        population_threshold, rng)

    A = np.column_stack([_unfold(block, diagonal_weight) for block in candidates])
    r = _unfold(rho_d, diagonal_weight)
    P, _ = nnls(A, r)

    keep = P > weight_threshold
    if not keep.any():
        keep = np.zeros_like(P, dtype=bool)
        keep[int(np.argmax(P))] = True  # degenerate fallback: keep the largest

    weights = P[keep]
    weights = weights / weights.sum()  # exact probability normalization for collapse
    blocks = np.array([candidates[a] for a in np.flatnonzero(keep)])
    return weights, blocks
