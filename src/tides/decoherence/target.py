import numpy as np

from tides.decoherence.decoherence_times import decoherence_time_matrix

'''
TAB target density matrix (plain exponential form).

Increment 2 of the rTAB-w1 implementation (see .claude/skills/tab-decoherence).
Given the fully coherent Ehrenfest density matrix at the end of a classical step,
rho^c, and the state-pairwise decoherence times tau_ij (increment 1), build the
target density matrix rho^d that the stochastic block collapse must reproduce in
the ensemble average.

Equations, transcribed from the typeset PDF -- Esch & Levine, J. Chem. Phys. 155,
214101 (2021), Eqs. 3-4 (identical to J. Chem. Phys. 152, 234105 (2020),
Eqs. 17-18):

    rho^d_ii = rho^c_ii(dt)                      (diagonal: population unchanged)
    rho^d_ij = rho^c_ij(dt) * exp(-dt / tau_ij)  (off-diagonal: coherence decayed)

The decay factor is real and positive, so each coherence keeps its complex phase
and only loses magnitude. Because tau_ij -> infinity for the diagonal and for
parallel-PES pairs, those elements are multiplied by exp(0) = 1, i.e. left intact.

This is the plain-exponential target; the TAB-w Gaussian-with-history target
(Paper C Eqs. 7-12) replaces the off-diagonal expression in a later increment,
leaving everything downstream (block decomposition, collapse) unchanged.
'''


def exponential_target(rho_c, tau, dt):
    '''Plain exponential TAB target density matrix rho^d (Eqs. 3-4).

    Parameters
    ----------
    rho_c : array_like, shape (nstates, nstates)
        Fully coherent Ehrenfest density matrix at the end of the classical step.
        Typically complex Hermitian (real populations on the diagonal, complex
        coherences off it), but any square matrix is accepted.
    tau : array_like, shape (nstates, nstates)
        State-pairwise decoherence times, e.g. from
        `decoherence_times.decoherence_time_matrix`. Diagonal and parallel-PES
        entries should be +inf; those coherences are then left unchanged.
    dt : float
        Classical time step (same unit system as tau).

    Returns
    -------
    rho_d : ndarray, shape (nstates, nstates)
        The target density matrix. Diagonal equals that of rho_c exactly (Eq. 3);
        off-diagonals are scaled by exp(-dt / tau_ij) (Eq. 4). Hermiticity and the
        total population (trace) of rho_c are preserved.
    '''
    rho_c = np.asarray(rho_c)
    tau = np.asarray(tau, dtype=float)
    if rho_c.ndim != 2 or rho_c.shape[0] != rho_c.shape[1]:
        raise ValueError(f'rho_c must be a square matrix; got shape {rho_c.shape}')
    if tau.shape != rho_c.shape:
        raise ValueError(f'tau shape {tau.shape} does not match rho_c shape {rho_c.shape}')

    # exp(-dt/tau): tau=+inf -> factor 1 (unchanged); tau->0 -> factor 0 (killed).
    decay = np.exp(-dt / tau)
    rho_d = rho_c * decay

    # Eq. 3: the diagonal is the coherent population, held fixed regardless of the
    # tau diagonal. This makes the contract match the equations exactly and keeps
    # the trace (total population) conserved.
    diag = np.diag_indices_from(rho_d)
    rho_d[diag] = np.diagonal(rho_c)
    return rho_d


def build_exponential_target(rho_c, favg, alpha, dt, hbar=1.0):
    '''Convenience: decoherence times + plain exponential target in one call.

    Chains `decoherence_time_matrix` (Eqs. 1-2) and `exponential_target`
    (Eqs. 3-4) for the common case where per-state time-averaged forces are on
    hand. Returns the target density matrix rho^d.

    Parameters
    ----------
    rho_c : array_like, shape (nstates, nstates)
        Coherent Ehrenfest density matrix at the end of the step.
    favg : array_like, shape (nstates, ndof)
        Time-averaged adiabatic forces (see `time_averaged_force`).
    alpha : float or array_like, shape (ndof,)
        Gaussian-width decoherence parameter(s), inverse length squared.
    dt : float
        Classical time step.
    hbar : float, optional
        Reduced Planck constant (default 1.0, atomic units).

    Returns
    -------
    ndarray, shape (nstates, nstates)
        The plain-exponential target density matrix rho^d.
    '''
    tau = decoherence_time_matrix(favg, alpha, hbar=hbar)
    return exponential_target(rho_c, tau, dt)
