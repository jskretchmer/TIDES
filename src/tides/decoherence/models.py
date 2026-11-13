import numpy as np

from tides.decoherence.decoherence_times import decoherence_time_matrix
from tides.decoherence.target import exponential_target
from tides.decoherence.blocks import greedy_block_decomposition
from tides.decoherence.rtab import rtab_decomposition
from tides.decoherence.collapse import stochastic_collapse, adiabatic_energy
from tides.decoherence.rescale import rescale_velocity

'''
Paper A model problems for validating the TAB collapse machinery.

Esch & Levine, J. Chem. Phys. 152, 234105 (2020), Sec. II C: one-dimensional,
uncoupled, linear adiabatic PESs. State i has energy E_i(x) = slope_i * x +
intercept_i, hence a constant force F_i = -slope_i and a constant state-pairwise
decoherence time. Because the states are uncoupled, the coherent populations are
constant and coherences only lose magnitude through the stochastic collapse -- so
the ensemble-averaged coherence should follow the exponential target of Eq. 29,

    |rho_ij(t)| = |rho_ij(0)| exp(-t / tau_ij).

This module provides the model definitions and a specialized propagator that drives
one trajectory (and ensembles) through the full increment 1-6 pipeline.
'''

# Shared parameters (Sec. II C).
MASS = 22500.0     # au, ~carbon
ALPHA = 4.7        # bohr^-2, ~ground-state vibrational width of H
DT = 2.5           # atu classical time step
X0 = 0.0           # initial position
PREFACTOR = 1.0    # Paper A tau convention (hbar^2 alpha, no factor of 8)

# Model definitions: slopes (hartree/bohr), intercepts (hartree), initial
# populations, initial momentum (au), propagation time (atu). Tables I-III.
MODELS = {
    'two_state': {
        'slopes': np.array([-0.015, 0.000]),
        'intercepts': np.array([0.020, 0.030]),
        'populations': np.array([0.25, 0.75]),
        'momentum': 30.000,
        'max_time': 1000.0,
    },
    'three_state': {
        'slopes': np.array([-0.030, 0.000, 0.000]),
        'intercepts': np.array([0.050, 0.060, 0.070]),
        'populations': np.array([0.50, 0.25, 0.25]),
        'momentum': 36.742,
        'max_time': 1000.0,
    },
    'five_state': {
        'slopes': np.array([-0.040, -0.030, -0.020, -0.010, 0.000]),
        'intercepts': np.array([0.030, 0.040, 0.050, 0.060, 0.070]),
        'populations': np.array([0.20, 0.20, 0.20, 0.20, 0.20]),
        'momentum': 47.434,
        'max_time': 1200.0,
    },
}


def model_decoherence_times(model, alpha=ALPHA, prefactor=PREFACTOR):
    '''State-pairwise tau_ij for a model (constant, from the linear-PES forces).'''
    forces = -np.asarray(model['slopes'])          # F_i = -slope_i
    favg = forces[:, np.newaxis]                    # one nuclear DOF
    return decoherence_time_matrix(favg, alpha, prefactor=prefactor)


def run_trajectory(model, rng, decomposition='greedy', alpha=ALPHA, dt=DT,
                   prefactor=PREFACTOR, rescale=True, record_stride=1):
    '''Propagate one trajectory of an uncoupled linear-PES model with TAB.

    Parameters
    ----------
    model : dict
        One of MODELS (or the same structure).
    rng : numpy.random.Generator
        Randomness for collapse and block pruning.
    decomposition : {'greedy', 'rtab'}
        Which decomposer to use for the collapse probabilities.
    alpha, dt, prefactor : float
        Decoherence width, classical time step, and tau convention.
    rescale : bool
        Whether to apply the energy-conserving velocity rescale after collapse.
    record_stride : int
        Record the density matrix every `record_stride` steps.

    Returns
    -------
    times : ndarray, shape (nrec,)
        Times at which the density matrix was recorded.
    rhos : ndarray, shape (nrec, nstates, nstates)
        Coherent density matrix rho^c at each recorded time (before that step's
        collapse; the diagonal is the trajectory's current populations).
    final_pops : ndarray, shape (nstates,)
        Populations at the end of the trajectory.
    '''
    slopes = np.asarray(model['slopes'])
    intercepts = np.asarray(model['intercepts'])
    forces = -slopes
    n = len(slopes)

    tau = decoherence_time_matrix(forces[:, None], alpha, prefactor=prefactor)
    decompose = greedy_block_decomposition if decomposition == 'greedy' else rtab_decomposition

    x = X0
    v = model['momentum'] / MASS
    c = np.sqrt(np.asarray(model['populations'], dtype=float)).astype(complex)

    nsteps = int(round(model['max_time'] / dt))
    times, rhos = [], []
    for step in range(nsteps):
        rho_c = np.outer(c, c.conj())
        if step % record_stride == 0:
            times.append(step * dt)
            rhos.append(rho_c)

        # Coherent Ehrenfest propagation over dt. Populations are constant (uncoupled),
        # so the mean-field force is constant over the step.
        pops = np.abs(c) ** 2
        if pops.max() > 1.0 - 1e-12:
            # Already collapsed to a single adiabatic state: no coherences remain, so
            # the decoherence correction is a no-op. Coast (phase only) for speed.
            f_mf = forces[int(pops.argmax())]
            x = x + v * dt + 0.5 * (f_mf / MASS) * dt ** 2
            v = v + (f_mf / MASS) * dt
            c = c * np.exp(-1j * (slopes * x + intercepts) * dt)
            continue
        f_mf = np.sum(pops * forces)
        x = x + v * dt + 0.5 * (f_mf / MASS) * dt ** 2
        v = v + (f_mf / MASS) * dt
        energies = slopes * x + intercepts
        c = c * np.exp(-1j * energies * dt)          # phases only; |c_i| unchanged

        # Decoherence correction: target -> decompose -> stochastic collapse.
        rho_c = np.outer(c, c.conj())
        rho_d = exponential_target(rho_c, tau, dt)
        weights, blocks = decompose(rho_d, rho_c, rng=rng)
        a, c_new = stochastic_collapse(c, weights, blocks, rng=rng)

        if rescale:
            de = adiabatic_energy(blocks[a], energies) - adiabatic_energy(rho_c, energies)
            v_arr, frustrated = rescale_velocity(np.array([v]), np.array([MASS]), de)
            if not frustrated:
                v = float(v_arr[0])
        c = c_new

    return np.array(times), np.array(rhos), np.abs(c) ** 2


def run_ensemble(model, ntraj, seed=0, decomposition='greedy', alpha=ALPHA, dt=DT,
                 prefactor=PREFACTOR, rescale=True, record_stride=1):
    '''Run an ensemble and return times, ensemble-averaged |rho_ij(t)|, final pops.

    Returns
    -------
    times : ndarray, shape (nrec,)
    mean_abs_rho : ndarray, shape (nrec, nstates, nstates)
        Ensemble average of |rho_ij(t)| (Eq. 30).
    final_pops : ndarray, shape (ntraj, nstates)
        Final populations of every trajectory (for Table IV style analysis).
    '''
    master = np.random.default_rng(seed)
    seeds = master.integers(0, 2 ** 63 - 1, size=ntraj)
    abs_rho_sum = None
    final_pops = []
    times = None
    for s in seeds:
        t, rhos, fpops = run_trajectory(
            model, np.random.default_rng(s), decomposition=decomposition,
            alpha=alpha, dt=dt, prefactor=prefactor, rescale=rescale,
            record_stride=record_stride)
        if abs_rho_sum is None:
            abs_rho_sum = np.abs(rhos)
            times = t
        else:
            abs_rho_sum += np.abs(rhos)
        final_pops.append(fpops)
    return times, abs_rho_sum / ntraj, np.array(final_pops)


def collapse_fractions(final_pops, threshold=0.98):
    '''Fraction of trajectories fully collapsed to each single state (Table IV).

    A trajectory has collapsed to state i if its final population on i exceeds
    `threshold` (0.98 in Paper A).
    '''
    final_pops = np.asarray(final_pops)
    return np.mean(final_pops > threshold, axis=0)
