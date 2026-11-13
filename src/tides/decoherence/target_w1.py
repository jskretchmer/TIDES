import numpy as np
from scipy.special import erf, erfc

from tides.decoherence.decoherence_times import decoherence_time_matrix

'''
TAB-w1 target density matrix (Gaussian coherence decay with w1 population history).

Increment 5 of the rTAB-w1 implementation (see .claude/skills/tab-decoherence). This
is a drop-in replacement for the plain-exponential off-diagonal target of increment 2:
the diagonal and everything downstream (block decomposition, collapse) are unchanged.

Esch & Levine, J. Chem. Phys. 155, 214101 (2021):

  Coherence formed at past time t' decays as a Gaussian in the Bittner-Rossky width
  (Eq. 7):
      G_ij(t, t') = exp( -(t - t')^2 / tau_ij^2 )

  Averaging over a formation-time history w_ij(t') (Eqs. 8-10), the off-diagonal
  target is (Eq. 9, with Gdot_ij = d/dt G_ij):
      rho^d_ij = rho^c_ij(dt) * [ 1 + ( INT w_ij(t') INT_0^dt Gdot_ij(t,t') dt dt' )
                                      / ( INT w_ij(t') G_ij(0,t') dt' ) ]

  The w1 history (Eqs. 11-12) is a step function built entirely from present-step
  quantities -- no stored history:
      w1_ij(t') = rhodot^c_kk / rho^c_kk   for t' in (-rho^c_kk/rhodot^c_kk, 0],
                                           provided rhodot^c_kk > 0
      w1_ij(t') = 0                        otherwise
      k = i if rhodot^c_ii > rhodot^c_jj else j

Closed form. Because INT_0^dt Gdot dt = G(dt,t') - G(0,t'), the bracket in Eq. 9
collapses to A/B with A = INT w G(dt,t') dt', B = INT w G(0,t') dt'. For the constant
step-function w1 the prefactor cancels, and the remaining Gaussian integrals give
error functions:

      rho^d_ij = rho^c_ij * [ erf((T + dt)/tau_ij) - erf(dt/tau_ij) ] / erf(T/tau_ij)

with T = rho^c_kk / rhodot^c_kk > 0. Limiting cases:
      T -> 0   (coherence just formed) -> exp(-dt^2/tau_ij^2)   (pure Gaussian)
      T -> inf (old coherence)         -> erfc(dt/tau_ij)
      tau_ij -> inf (parallel / self)  -> 1                     (no decoherence)

Edge case (paper is silent): when rhodot^c_kk <= 0 the w1 history vanishes and Eq. 9
is 0/0. We fall back to the T -> inf limit, erfc(dt/tau_ij), which is the continuous
continuation from rhodot^c_kk -> 0+. This choice is exposed via `no_history_decay`.
'''

# Below this magnitude of erf(T/tau) we use the T -> 0 Gaussian limit to avoid 0/0.
_SMALL_ERF = 1e-12


def tabw1_decay_factor(tau, dt, pops, pop_deriv, no_history_decay='erfc'):
    '''Per-pair TAB-w1 coherence-survival factors (Eq. 9, closed form).

    Parameters
    ----------
    tau : array_like, shape (nstates, nstates)
        State-pairwise decoherence times (from `decoherence_time_matrix`); +inf on
        the diagonal and for parallel-PES pairs.
    dt : float
        Classical time step.
    pops : array_like, shape (nstates,)
        Coherent populations rho^c_ii(dt).
    pop_deriv : array_like, shape (nstates,)
        Time derivatives rhodot^c_ii(dt) of the coherent populations.
    no_history_decay : {'erfc', 'one'}, optional
        Fallback when rhodot^c_kk <= 0 (w1 history empty): 'erfc' (default) uses
        erfc(dt/tau), the continuous limit; 'one' leaves the coherence unchanged.

    Returns
    -------
    decay : ndarray, shape (nstates, nstates)
        Symmetric matrix of survival factors in (0, 1]; multiply rho^c off-diagonals
        by these to obtain the TAB-w1 target.
    '''
    tau = np.asarray(tau, dtype=float)
    pops = np.asarray(pops, dtype=float)
    pdots = np.asarray(pop_deriv, dtype=float)
    n = tau.shape[0]

    # k = faster-changing state of each pair (Eq. 12); gather its population and rate.
    faster_i = pdots[:, None] > pdots[None, :]
    k_pdot = np.maximum(pdots[:, None], pdots[None, :])              # rhodot^c_kk
    k_pop = np.where(faster_i, pops[:, None], pops[None, :])         # rho^c_kk

    finite = np.isfinite(tau)
    decay = np.ones((n, n))  # default 1 covers tau=inf (parallel/self) and diagonal

    # Fallback region: finite tau but empty w1 history (rhodot^c_kk <= 0).
    empty_hist = finite & (k_pdot <= 0.0)
    if no_history_decay == 'erfc':
        decay[empty_hist] = erfc(dt / tau[empty_hist])
    elif no_history_decay == 'one':
        decay[empty_hist] = 1.0
    else:
        raise ValueError("no_history_decay must be 'erfc' or 'one'")

    # Main region: finite tau and rhodot^c_kk > 0.
    main = finite & (k_pdot > 0.0)
    tau_m = tau[main]
    T = k_pop[main] / k_pdot[main]                                   # formation timescale
    den = erf(T / tau_m)
    small = den < _SMALL_ERF                                         # T -> 0 limit
    den_safe = np.where(small, 1.0, den)
    num = erf((T + dt) / tau_m) - erf(dt / tau_m)
    decay[main] = np.where(small, np.exp(-(dt ** 2) / tau_m ** 2), num / den_safe)
    return decay


def tabw1_target(rho_c, tau, dt, pop_deriv, no_history_decay='erfc'):
    '''TAB-w1 target density matrix rho^d (Eq. 9, closed form).

    Sibling of `target.exponential_target`: off-diagonal replacement only. The
    diagonal equals that of rho_c (populations unchanged), so Hermiticity and the
    total population are preserved.

    Parameters
    ----------
    rho_c : array_like, shape (nstates, nstates)
        Fully coherent Ehrenfest density matrix at the end of the classical step.
    tau : array_like, shape (nstates, nstates)
        State-pairwise decoherence times.
    dt : float
        Classical time step.
    pop_deriv : array_like, shape (nstates,)
        Time derivatives rhodot^c_ii(dt) of the coherent populations.
    no_history_decay : {'erfc', 'one'}, optional
        Fallback when a pair's w1 history is empty (see `tabw1_decay_factor`).

    Returns
    -------
    rho_d : ndarray, shape (nstates, nstates)
        The TAB-w1 target density matrix.
    '''
    rho_c = np.asarray(rho_c)
    if rho_c.ndim != 2 or rho_c.shape[0] != rho_c.shape[1]:
        raise ValueError(f'rho_c must be a square matrix; got shape {rho_c.shape}')
    tau = np.asarray(tau, dtype=float)
    if tau.shape != rho_c.shape:
        raise ValueError(f'tau shape {tau.shape} does not match rho_c shape {rho_c.shape}')

    pops = np.diagonal(rho_c).real
    decay = tabw1_decay_factor(tau, dt, pops, pop_deriv, no_history_decay=no_history_decay)
    rho_d = rho_c * decay
    diag = np.diag_indices_from(rho_d)
    rho_d[diag] = np.diagonal(rho_c)
    return rho_d


def build_tabw1_target(rho_c, favg, alpha, dt, pop_deriv, hbar=1.0,
                       no_history_decay='erfc'):
    '''Convenience: decoherence times + TAB-w1 target in one call.

    Chains `decoherence_time_matrix` (Eqs. 1-2) and `tabw1_target` (Eq. 9).

    Parameters
    ----------
    rho_c : array_like, shape (nstates, nstates)
        Coherent Ehrenfest density matrix at the end of the step.
    favg : array_like, shape (nstates, ndof)
        Time-averaged adiabatic forces.
    alpha : float or array_like, shape (ndof,)
        Gaussian-width decoherence parameter(s), inverse length squared.
    dt : float
        Classical time step.
    pop_deriv : array_like, shape (nstates,)
        Time derivatives of the coherent populations.
    hbar : float, optional
        Reduced Planck constant (default 1.0, atomic units).
    no_history_decay : {'erfc', 'one'}, optional
        Empty-history fallback (see `tabw1_decay_factor`).

    Returns
    -------
    ndarray, shape (nstates, nstates)
        The TAB-w1 target density matrix rho^d.
    '''
    tau = decoherence_time_matrix(favg, alpha, hbar=hbar)
    return tabw1_target(rho_c, tau, dt, pop_deriv, no_history_decay=no_history_decay)
