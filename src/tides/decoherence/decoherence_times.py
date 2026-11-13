import numpy as np

'''
State-pairwise decoherence times for TAB / TAB-w decoherence-corrected Ehrenfest.

Increment 1 of the rTAB-w1 implementation (see .claude/skills/tab-decoherence).
This module is deliberately standalone: it operates only on plain NumPy arrays of
adiabatic forces and takes no dependency on the electronic-structure stack, so the
collapse machinery can be developed and validated against the 1-D model problems
before being wired into an ab-initio method.

Canonical equations, transcribed from the typeset PDFs (not the OCR notes):

  Time-averaged adiabatic force -- Esch & Levine, J. Chem. Phys. 155, 214101 (2021),
  Eq. 2 (identical to J. Chem. Phys. 152, 234105 (2020), Eq. 7):

      F^avg_{i,eta} = [ F_{i,eta}(0) + F_{i,eta}(dt) ] / 2

  State-pairwise (Bittner-Rossky) decoherence time -- J. Chem. Phys. 155, 214101
  (2021), Eq. 1:

      1 / tau_ij^2 = sum_eta ( F^avg_{i,eta} - F^avg_{j,eta} )^2 / ( 8 hbar^2 alpha_eta )

  where eta indexes the classical nuclear degrees of freedom and alpha_eta (units of
  inverse length squared) is the Gaussian-width decoherence parameter.

  Prefactor convention: the factor of 8 is the Paper C (TAB-w) convention. Paper A
  (Eq. 13) writes the same quantity with 1/(hbar^2 alpha_eta) because alpha is defined
  a factor of two differently there (Paper C footnote to Eq. 1). The TiDES target
  method is rTAB-w1, i.e. Paper C, so the factor of 8 is the correct one to use here.

  Because tau_ij depends only on the *difference* of adiabatic forces, pairs of
  parallel PESs (equal forces) give tau_ij -> infinity: they never lose coherence.
  The self-time tau_ii is likewise infinite. This is the whole point of a
  state-pairwise (rather than state-wise) decoherence time.
'''


def time_averaged_force(force_t0, force_tdt):
    '''Trapezoidal average of the adiabatic force over one classical step (Eq. 2).

    Parameters
    ----------
    force_t0, force_tdt : array_like
        Adiabatic force at the start and end of the classical time step. May be a
        single force vector of shape (ndof,) or a stack of per-state forces of
        shape (nstates, ndof); any common broadcastable shape is accepted.

    Returns
    -------
    ndarray
        0.5 * (force_t0 + force_tdt), with the shape of the (broadcast) inputs.
    '''
    return 0.5 * (np.asarray(force_t0, dtype=float) + np.asarray(force_tdt, dtype=float))


def decoherence_time_matrix(favg, alpha, hbar=1.0, prefactor=8.0):
    '''State-pairwise decoherence-time matrix tau_ij (Eq. 1).

    Parameters
    ----------
    favg : array_like, shape (nstates, ndof)
        Time-averaged adiabatic forces, one row per adiabatic state (see
        `time_averaged_force`).
    alpha : float or array_like, shape (ndof,)
        Gaussian-width decoherence parameter(s), in inverse length squared. A
        scalar is broadcast to every classical degree of freedom.
    hbar : float, optional
        Reduced Planck constant in the working unit system (default 1.0, i.e.
        atomic units).
    prefactor : float, optional
        Denominator constant multiplying hbar^2 alpha_eta: 8.0 for the Paper C
        (TAB-w) convention -- the default and the rTAB-w1 target -- or 1.0 for the
        Paper A / Paper B convention (J. Chem. Phys. 152, 234105 (2020), Eq. 13),
        which differs only by the definition of the Gaussian width alpha. Use 1.0
        to reproduce the Paper A model-problem numbers.

    Returns
    -------
    tau : ndarray, shape (nstates, nstates)
        Symmetric matrix of state-pairwise decoherence times. Diagonal entries and
        entries for parallel-PES pairs (zero force difference) are +inf.
    '''
    favg = np.asarray(favg, dtype=float)
    if favg.ndim != 2:
        raise ValueError(f'favg must be 2-D (nstates, ndof); got shape {favg.shape}')
    nstates, ndof = favg.shape
    alpha = np.broadcast_to(np.asarray(alpha, dtype=float), (ndof,))
    if np.any(alpha <= 0):
        raise ValueError('alpha (decoherence width) must be positive for every DOF')

    # Pairwise force differences: dF[i, j, eta] = F^avg_{i,eta} - F^avg_{j,eta}
    dF = favg[:, np.newaxis, :] - favg[np.newaxis, :, :]
    inv_tau2 = np.sum(dF ** 2 / (prefactor * hbar ** 2 * alpha), axis=-1)

    with np.errstate(divide='ignore'):
        # inv_tau2 == 0 on the diagonal and for parallel PESs -> tau = +inf
        tau = 1.0 / np.sqrt(inv_tau2)
    return tau


def decoherence_time(favg_i, favg_j, alpha, hbar=1.0, prefactor=8.0):
    '''Scalar tau_ij for a single pair of adiabatic states (Eq. 1).

    Convenience wrapper around `decoherence_time_matrix` for one state pair.

    Parameters
    ----------
    favg_i, favg_j : array_like, shape (ndof,)
        Time-averaged adiabatic forces on states i and j.
    alpha : float or array_like, shape (ndof,)
        Gaussian-width decoherence parameter(s), inverse length squared.
    hbar : float, optional
        Reduced Planck constant (default 1.0, atomic units).
    prefactor : float, optional
        Convention constant (8.0 Paper C, 1.0 Paper A); see
        `decoherence_time_matrix`.

    Returns
    -------
    float
        The decoherence time tau_ij (+inf for parallel forces).
    '''
    favg = np.stack([np.asarray(favg_i, dtype=float),
                     np.asarray(favg_j, dtype=float)])
    return float(decoherence_time_matrix(favg, alpha, hbar=hbar, prefactor=prefactor)[0, 1])
