import numpy as np

from tides.decoherence.decoherence_times import decoherence_time_matrix
from tides.decoherence.target import exponential_target
from tides.decoherence.blocks import greedy_block_decomposition, robust_greedy_decomposition
from tides.decoherence.rtab import rtab_decomposition
from tides.decoherence.collapse import stochastic_collapse, adiabatic_energy
from tides.decoherence.rescale import rescale_velocity

'''
Paper B / C band models for validating TAB on coupled multi-state scattering.

Esch & Levine, J. Chem. Phys. 153, 114104 (2020), Sec. II C (Eqs. 19-23). Diabatic
electronic Hamiltonian: a single negatively-sloped "diabat 1",

    H_11 = -w1 * x,

and a band of M-1 positively-sloped diabats,

    H_ii = w2 * x - (i-1) * delta            (1 < i <= M),

each coupled to diabat 1 (and not to each other),

    H_1i = H_i1 = k,   H_ij = 0 (i != j, both in band).

In the "split" model the upper half of the band is shifted down by a gap eps. With
w1 = 0.25, w2 = 0.025, k = 0.005 a.u., a wave packet launched from x0 = -1.0 with
momentum 10.0 a.u. (mass 1822) scatters through the crossing; transmission is the
fraction of trajectories that end with >98% population on the lowest adiabatic state
(which is diabat 1 asymptotically).

Because the states are coupled, this driver diagonalizes H(x) to get adiabatic
energies/forces and propagates the electronic wavefunction in the smooth diabatic
basis, transforming to the adiabatic basis only for the TAB collapse.
'''

W1 = 0.25
W2 = 0.025
K = 0.005
MASS = 1822.0
X0 = -1.0
P0 = 10.0
DT = 0.10

_UHARTREE = 1e-6  # microhartree -> hartree

# Table I: (M, delta [microhartree], eps [microhartree] or None)
BAND_MODELS = {
    '9_10000':       {'M': 9,  'delta': 10000, 'eps': None},
    '9_10000_split': {'M': 9,  'delta': 10000, 'eps': 80000},
    '17_5000':       {'M': 17, 'delta': 5000,  'eps': None},
    '17_2500':       {'M': 17, 'delta': 2500,  'eps': None},
    '17_1250':       {'M': 17, 'delta': 1250,  'eps': None},
    '17_625':        {'M': 17, 'delta': 625,   'eps': None},
}


def diabatic_diagonal_intercepts(model):
    '''Constant part of each diabatic diagonal, H_ii(x) - slope_i * x (Eqs. 20-22).'''
    M = model['M']
    delta = model['delta'] * _UHARTREE
    eps = model['eps']
    intercepts = np.zeros(M)
    intercepts[0] = 0.0  # diabat 1: H_11 = -w1 x
    for j in range(1, M):        # 0-based band index j = (1-based i) - 1
        intercepts[j] = -j * delta
    if eps is not None:
        eps_h = eps * _UHARTREE
        # upper half of the band (1-based i > (M-1)/2 + 1) shifted down by eps
        first_upper = (M - 1) // 2 + 1  # 0-based first index of the upper sub-band
        intercepts[first_upper:] -= eps_h
    return intercepts


def diabatic_slopes(model):
    '''Slope of each diabatic diagonal: -w1 for diabat 1, +w2 for the band.'''
    M = model['M']
    slopes = np.full(M, W2)
    slopes[0] = -W1
    return slopes


def hamiltonian(x, model, slopes=None, intercepts=None):
    '''Diabatic electronic Hamiltonian H(x) (M x M, real symmetric).'''
    M = model['M']
    if slopes is None:
        slopes = diabatic_slopes(model)
    if intercepts is None:
        intercepts = diabatic_diagonal_intercepts(model)
    H = np.zeros((M, M))
    H[np.diag_indices(M)] = slopes * x + intercepts
    H[0, 1:] = K
    H[1:, 0] = K
    return H


def adiabatic(x, model, slopes=None, intercepts=None):
    '''Adiabatic energies (ascending) and states U (columns) at position x.'''
    H = hamiltonian(x, model, slopes, intercepts)
    energies, U = np.linalg.eigh(H)
    return energies, U


def adiabatic_forces(U, slopes):
    '''Hellmann-Feynman adiabatic forces F_a = -<a| dH/dx |a>.

    dH/dx = diag(slopes) is constant, so F_a = -sum_i U[i,a]^2 * slope_i.
    '''
    return -np.einsum('ia,i->a', U ** 2, slopes)


def run_trajectory(model, rng, alpha=216.0, dt=DT, max_time=400.0, prefactor=1.0,
                   decomposition='rtab', rescale=True, coherence_tol=1e-6):
    '''Propagate one coupled-band trajectory with TAB decoherence.

    Returns
    -------
    pop_lowest_adiabat : float
        Final population on the lowest adiabatic state (transmission indicator).
    pop_diabat1 : float
        Final population on diabat 1 (for comparison with exact quantum results).
    '''
    slopes = diabatic_slopes(model)
    intercepts = diabatic_diagonal_intercepts(model)
    dHdx = slopes  # diagonal of dH/dx
    decompose = greedy_block_decomposition if decomposition == 'greedy' else rtab_decomposition

    x = X0
    v = P0 / MASS
    # start on diabat 1 (a Gaussian on diabat 1 in the exact picture)
    c = np.zeros(model['M'], dtype=complex)
    c[0] = 1.0

    energies, U = adiabatic(x, model, slopes, intercepts)
    f_adia = adiabatic_forces(U, slopes)
    f_mf = -np.real(c.conj() @ (dHdx * c))

    nsteps = int(round(max_time / dt))
    for _ in range(nsteps):
        # --- nuclear position (velocity Verlet, first half) ---
        x = x + v * dt + 0.5 * (f_mf / MASS) * dt ** 2
        energies, U = adiabatic(x, model, slopes, intercepts)
        f_adia_new = adiabatic_forces(U, slopes)

        # --- electronic propagation in the diabatic basis (frozen H over the step) ---
        c_ad = U.conj().T @ c
        c_ad = np.exp(-1j * energies * dt) * c_ad
        c = U @ c_ad
        f_mf_new = -np.real(c.conj() @ (dHdx * c))

        # --- nuclear velocity (velocity Verlet, second half) ---
        v = v + 0.5 * (f_mf + f_mf_new) / MASS * dt
        f_mf = f_mf_new

        # --- TAB decoherence correction in the adiabatic basis ---
        c_ad = U.conj().T @ c
        rho_ad = np.outer(c_ad, c_ad.conj())
        offdiag = np.abs(rho_ad - np.diag(np.diagonal(rho_ad)))
        if offdiag.max() > coherence_tol:  # skip the collapse when fully decohered
            favg = 0.5 * (f_adia + f_adia_new)
            tau = decoherence_time_matrix(favg[:, None], alpha, prefactor=prefactor)
            rho_d = exponential_target(rho_ad, tau, dt)
            if decomposition == 'rtab':
                weights, blocks = rtab_decomposition(rho_d, rho_ad, rng=rng)
            else:  # 'greedy' -> robust terminating greedy (fast, clipped weights)
                weights, blocks = robust_greedy_decomposition(rho_d, rho_ad, rng=rng)
            a, c_ad_new = stochastic_collapse(c_ad, weights, blocks, rng=rng)
            if rescale:
                de = adiabatic_energy(blocks[a], energies) - adiabatic_energy(rho_ad, energies)
                v_arr, frustrated = rescale_velocity(np.array([v]), np.array([MASS]), de)
                if not frustrated:
                    v = float(v_arr[0])
            c = U @ c_ad_new
            f_mf = -np.real(c.conj() @ (dHdx * c))

        f_adia = f_adia_new

    c_ad = U.conj().T @ c
    return float(np.abs(c_ad[0]) ** 2), float(np.abs(c[0]) ** 2)


def transmission(model, ntraj, seed=0, alpha=216.0, dt=DT, max_time=400.0,
                 prefactor=1.0, decomposition='rtab', rescale=True, threshold=0.98):
    '''Transmission probability: fraction of trajectories with >threshold population
    on the lowest adiabatic state at the final time (Paper B definition).

    Returns (transmission_prob, mean_diabat1_population).
    '''
    master = np.random.default_rng(seed)
    seeds = master.integers(0, 2 ** 63 - 1, size=ntraj)
    pop_low = np.empty(ntraj)
    pop_dia1 = np.empty(ntraj)
    for i, s in enumerate(seeds):
        pop_low[i], pop_dia1[i] = run_trajectory(
            model, np.random.default_rng(s), alpha=alpha, dt=dt, max_time=max_time,
            prefactor=prefactor, decomposition=decomposition, rescale=rescale)
    return float(np.mean(pop_low > threshold)), float(np.mean(pop_dia1))
