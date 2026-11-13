import numpy as np

'''
Greedy block decomposition -- the heart of the TAB collapse.

Increment 3 of the rTAB-w1 implementation (see .claude/skills/tab-decoherence).
Given the target density matrix rho^d (increment 2) and the fully coherent Ehrenfest
density matrix rho^c, write rho^d as a weighted sum of "block" density matrices,

    rho^d = sum_a P^block_a rho^block_a                    (B Eq. 5 / A Eq. 19)

where each rho^block_a is a coherent superposition over a subset beta_a of the
populated adiabatic states, normalized to unit trace, and the weights P^block_a are
non-negative and sum to one (they become the collapse probabilities in increment 4).

Algorithm (Esch & Levine, J. Chem. Phys. 153, 114104 (2020), Eqs. 11-16; cross-checked
against J. Chem. Phys. 152, 234105 (2020), Eqs. 19-25). rho^tmp starts as a copy of
rho^d and is the running residual. Looping with block index a:

  1. b_ij = rho^tmp_ij / rho^c_ij                          (B Eq. 11)
     (structurally-zero coherences, rho^c_ij == 0, give b_ij := 0.)
  2. Find the smallest non-zero element b_kl (|b| > threshold). It may be
     off-diagonal (k != l) or diagonal (k == l).           (B step 2)
  3. Grow the block set beta from the seed {k, l}:
       - add state p (p != k, l) if b_pl != 0 and b_pk != 0 (B Eq. 12);
       - for pairs r, s already in beta with b_rs == 0, randomly drop one
         (B Eq. 13); once a state is dropped, pairs involving it are skipped.
  4. rho^block_{a,mn} = rho^c_mn / sum_{i in beta} rho^c_ii for m, n in beta;
     zero otherwise.                                        (B Eq. 14)
  5. P^block_a = rho^tmp_kl / rho^block_{a,kl}; clamp small negatives
     (-1e-7 < P < 0) to exactly 0.                          (B Eq. 15)
  6. rho^tmp <- rho^tmp - P^block_a rho^block_a             (B Eq. 16)
  Repeat until every element of b is numerically zero. Each iteration zeros at
  least one (upper-triangle) element of b, so this terminates in at most
  N(N+1)/2 iterations.

The decomposition is not unique (the randomized pruning is why rTAB, increment 6,
later replaces this greedy weighting with non-negative least squares for the
multi-dimensional case). Reconstruction sum_a P_a rho^block_a = rho^d holds for any
valid random choices, and is the primary correctness check.
'''


def _seed_index(b, threshold):
    '''Return (k, l) of the smallest non-zero element of b, or None if all zero.

    "Non-zero" means |b_ij| > threshold. Among those, the smallest by (real,
    signed) value is chosen; for the physically expected non-negative b this is the
    smallest magnitude. On the first iteration b holds the coherence-survival
    factors exp(-dt/tau_ij) in (0, 1] with a unit diagonal, so the smallest such
    element is the fastest-decohering pair.
    '''
    nonzero = np.abs(b) > threshold
    if not nonzero.any():
        return None
    masked = np.where(nonzero, b, np.inf)
    k, l = np.unravel_index(np.argmin(masked), b.shape)
    return int(k), int(l)


def _prune_incoherent_pairs(beta, k, l, b, threshold, rng):
    '''Drop states so that no surviving pair in beta has zero mutual coherence.

    Implements B Eq. 13: for pairs r, s in beta (excluding the seed states k, l,
    which are mutually coherent by construction), if b_rs == 0 then r or s is
    removed at random; once a state is removed, further pairs involving it are not
    considered.
    '''
    extra = [p for p in sorted(beta) if p != k and p != l]
    removed = set()
    for i, r in enumerate(extra):
        if r in removed:
            continue
        for s in extra[i + 1:]:
            if s in removed:
                continue
            if abs(b[r, s]) <= threshold:  # b_rs == 0: r and s cannot share a block
                if rng.random() < 0.5:
                    removed.add(r)
                    break  # r is gone; stop pairing it
                removed.add(s)
    return {p for p in beta if p not in removed}


def greedy_block_decomposition(rho_d, rho_c, threshold=1e-9, neg_weight_tol=1e-7,
                               coherence_threshold=1e-12, rng=None):
    '''Greedy decomposition of rho^d into weighted block density matrices.

    Parameters
    ----------
    rho_d : array_like, shape (nstates, nstates)
        Target density matrix (e.g. from `target.exponential_target`). Its diagonal
        must equal that of rho_c (the TAB target leaves populations unchanged).
    rho_c : array_like, shape (nstates, nstates)
        Fully coherent Ehrenfest density matrix at the end of the classical step.
        Assumed Hermitian and positive-semidefinite with unit trace.
    threshold : float, optional
        Magnitude below which an element of b is treated as zero (default 1e-9, the
        Paper B value).
    neg_weight_tol : float, optional
        Small negative weights in (-neg_weight_tol, 0) are clamped to 0 (default
        1e-7, the Paper B value) to absorb numerical noise.
    coherence_threshold : float, optional
        Magnitude below which an element of rho^c is treated as structurally zero
        (default 1e-12).
    rng : numpy.random.Generator, optional
        Source of randomness for the pruning step. Defaults to a fresh default_rng.

    Returns
    -------
    weights : ndarray, shape (nblocks,)
        The block weights P^block_a (non-negative; sum to 1 for a valid target).
    blocks : ndarray, shape (nblocks, nstates, nstates)
        The block density matrices rho^block_a, each Hermitian with unit trace and
        support on its state subset beta_a.
    '''
    rho_d = np.asarray(rho_d)
    rho_c = np.asarray(rho_c)
    if rho_d.ndim != 2 or rho_d.shape[0] != rho_d.shape[1]:
        raise ValueError(f'rho_d must be a square matrix; got shape {rho_d.shape}')
    if rho_c.shape != rho_d.shape:
        raise ValueError(f'rho_c shape {rho_c.shape} does not match rho_d shape {rho_d.shape}')
    n = rho_d.shape[0]
    if rng is None:
        rng = np.random.default_rng()

    rho_c = rho_c.astype(complex)
    rho_tmp = np.array(rho_d, dtype=complex)  # running residual (starts as rho^d)
    pops_c = np.diagonal(rho_c).real          # rho^c_ii
    exists = np.abs(rho_c) > coherence_threshold  # which coherences physically exist

    weights = []
    blocks = []
    max_iter = n * (n + 1) // 2 + 5  # theoretical bound N(N+1)/2, plus a margin

    for _ in range(max_iter):
        # (Eq. 11) b_ij = rho^tmp_ij / rho^c_ij; structurally-zero coherences -> 0.
        # b is real because rho^tmp_ij shares rho^c_ij's complex phase throughout.
        b = np.zeros((n, n))
        b[exists] = (rho_tmp[exists] / rho_c[exists]).real

        seed = _seed_index(b, threshold)
        if seed is None:
            break  # every element of b is numerically zero: decomposition complete
        k, l = seed

        # (Eq. 12) grow beta with states coherent with both seed states
        beta = {k, l}
        for p in range(n):
            if p != k and p != l and abs(b[p, l]) > threshold and abs(b[p, k]) > threshold:
                beta.add(p)
        # (Eq. 13) prune pairs whose mutual coherence has already been resolved
        beta = _prune_incoherent_pairs(beta, k, l, b, threshold, rng)
        beta_list = sorted(beta)

        # (Eq. 14) block density matrix on beta, normalized by its population sum
        pop_sum = pops_c[beta_list].sum()
        block = np.zeros((n, n), dtype=complex)
        sub = np.ix_(beta_list, beta_list)
        block[sub] = rho_c[sub] / pop_sum

        # (Eq. 15) weight; clamp tiny negatives from numerical noise
        P = (rho_tmp[k, l] / block[k, l]).real
        clamped = -neg_weight_tol < P < 0.0
        if clamped:
            P = 0.0

        # (Eq. 16) remove this block's contribution from the residual
        rho_tmp = rho_tmp - P * block
        if clamped:
            # P was clamped to 0, so the seed element was not zeroed by the
            # subtraction; discard the negligible residual coherence to guarantee
            # progress (and thus termination).
            rho_tmp[k, l] = 0.0
            rho_tmp[l, k] = 0.0

        weights.append(P)
        blocks.append(block)
    else:
        raise RuntimeError('greedy block decomposition failed to converge; residual '
                           f'coherence remains after {max_iter} iterations')

    if not blocks:
        return np.zeros(0), np.zeros((0, n, n), dtype=complex)
    return np.array(weights), np.array(blocks)


def robust_greedy_decomposition(rho_d, rho_c, threshold=1e-9, neg_weight_tol=1e-7,
                                coherence_threshold=1e-12, rng=None):
    '''Terminating greedy decomposition with clipped, renormalized weights.

    Like `greedy_block_decomposition`, but marks each seed position resolved so it is
    never revisited. This guarantees termination in at most N(N+1)/2 iterations even
    for the near-degenerate / many-parallel-state targets (e.g. the band models)
    where the exact greedy would fail to converge, and it clips the resulting weights
    to valid non-negative collapse probabilities. Faster than rTAB (no least-squares
    solve); use it when exact greedy would hang or go negative and NNLS is too costly.
    The trade-off is that the reconstruction is approximate on those pathological
    targets (exact greedy is exact where it converges).

    Returns (weights, blocks) in the same form as `greedy_block_decomposition`, with
    non-negative weights summing to 1.
    '''
    rho_d = np.asarray(rho_d)
    rho_c = np.asarray(rho_c)
    if rho_d.ndim != 2 or rho_d.shape[0] != rho_d.shape[1]:
        raise ValueError(f'rho_d must be a square matrix; got shape {rho_d.shape}')
    if rho_c.shape != rho_d.shape:
        raise ValueError(f'rho_c shape {rho_c.shape} does not match rho_d shape {rho_d.shape}')
    n = rho_d.shape[0]
    if rng is None:
        rng = np.random.default_rng()

    rho_c = rho_c.astype(complex)
    rho_tmp = np.array(rho_d, dtype=complex)
    pops_c = np.diagonal(rho_c).real
    exists = np.abs(rho_c) > coherence_threshold
    resolved = np.zeros((n, n), dtype=bool)

    weights, blocks = [], []
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

        pop_sum = pops_c[beta_list].sum()
        block = np.zeros((n, n), dtype=complex)
        sub = np.ix_(beta_list, beta_list)
        block[sub] = rho_c[sub] / pop_sum

        P = (rho_tmp[k, l] / block[k, l]).real
        if -neg_weight_tol < P < 0.0:
            P = 0.0
        rho_tmp = rho_tmp - P * block
        resolved[k, l] = resolved[l, k] = True

        weights.append(P)
        blocks.append(block)

    if not blocks:
        return np.zeros(0), np.zeros((0, n, n), dtype=complex)
    weights = np.clip(np.array(weights), 0.0, None)
    total = weights.sum()
    if total > 0:
        weights = weights / total
    return weights, np.array(blocks)
