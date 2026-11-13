import numpy as np

'''
Energy-conserving nuclear momentum rescale after a collapse.

Increment 4 (part 2) of the rTAB-w1 implementation (see .claude/skills/tab-decoherence).
Collapsing the electronic wavefunction onto a block changes the electronic
(Ehrenfest) energy, so the nuclear velocities are rescaled to keep the total energy
constant (Esch & Levine, J. Chem. Phys. 152, 234105 (2020), text after Eq. 28; and
J. Chem. Phys. 155, 214101 (2021), Sec. II B).

Total energy is E_kin + E_elec. If the collapse changes the electronic energy by
delta_energy = E_elec(after) - E_elec(before), the nuclear kinetic energy must change
by -delta_energy.

Two open questions from the 1-D papers, exposed here as explicit knobs (as the skill
guardrails require):

  * Rescale direction. In 1-D the momentum is a scalar and rescaling is unambiguous:
    scale the whole velocity vector (`direction=None`). In multi-D the papers defer;
    the recommended default for TiDES is to rescale along the nonadiabatic-coupling
    (NAC) direction (`direction=<NAC vector>`), as in fewest-switches surface hopping.
    The two are equivalent in 1-D.

  * Frustrated (energy-forbidden) collapses. When there is not enough kinetic energy
    to absorb an increase in electronic energy, the rescale is impossible. The 1-D
    models never trigger this, so the papers implement no fix. Here the rescale simply
    reports `frustrated=True` and returns the velocity unchanged; the caller's policy
    (reject the collapse, reverse the velocity, ...) is left as a higher-level knob.
'''


def kinetic_energy(vel, mass):
    '''Nuclear kinetic energy 0.5 * sum(mass * vel**2).'''
    vel = np.asarray(vel, dtype=float)
    mass = np.asarray(mass, dtype=float)
    return 0.5 * float(np.sum(mass * vel ** 2))


def rescale_velocity(vel, mass, delta_energy, direction=None):
    '''Rescale velocities to change kinetic energy by -delta_energy.

    Parameters
    ----------
    vel : array_like
        Nuclear velocities (any shape; must match `mass`).
    mass : array_like
        Nuclear masses, same shape as `vel`.
    delta_energy : float
        Change in electronic energy across the collapse, E_elec(after) -
        E_elec(before). Kinetic energy is adjusted by -delta_energy.
    direction : array_like, optional
        Direction along which to rescale the momentum (e.g. the NAC vector), same
        shape as `vel`. If None (default), the whole velocity vector is scaled
        uniformly -- the unambiguous 1-D choice.

    Returns
    -------
    vel_new : ndarray
        The rescaled velocities (unchanged if the collapse is frustrated).
    frustrated : bool
        True if there was not enough kinetic energy to conserve total energy; the
        velocity is then returned unchanged.
    '''
    vel = np.asarray(vel, dtype=float)
    mass = np.asarray(mass, dtype=float)

    if direction is None:
        ke_old = 0.5 * np.sum(mass * vel ** 2)
        ke_new = ke_old - delta_energy
        if ke_new < 0.0 or ke_old == 0.0:
            # Not enough KE to give up (or no KE to scale at all).
            if delta_energy == 0.0:
                return vel.copy(), False
            return vel.copy(), True
        scale = np.sqrt(ke_new / ke_old)
        return scale * vel, False

    # Directional rescale: v_new = v - gamma * direction / mass, choosing gamma to
    # conserve total energy. This yields the quadratic
    #     a * gamma^2 - b * gamma + delta_energy = 0
    # with a = 0.5 * sum(direction^2 / mass), b = sum(vel * direction).
    direction = np.asarray(direction, dtype=float)
    a = 0.5 * np.sum(direction ** 2 / mass)
    b = np.sum(vel * direction)
    if a == 0.0:
        # Degenerate direction; fall back to no rescale, frustrated unless trivial.
        return vel.copy(), (delta_energy != 0.0)

    discriminant = b ** 2 - 4.0 * a * delta_energy
    if discriminant < 0.0:
        return vel.copy(), True  # frustrated: no real solution
    root = np.sqrt(discriminant)
    # Choose the solution with the smaller displacement (standard surface hopping).
    gamma_plus = (b + root) / (2.0 * a)
    gamma_minus = (b - root) / (2.0 * a)
    gamma = gamma_minus if abs(gamma_minus) <= abs(gamma_plus) else gamma_plus
    return vel - gamma * direction / mass, False
