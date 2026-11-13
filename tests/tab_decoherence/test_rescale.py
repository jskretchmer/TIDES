import numpy as np
import pytest

from tides.decoherence import kinetic_energy, rescale_velocity

# Increment 4 (rescale) of the rTAB-w1 build (see .claude/skills/tab-decoherence).
# Energy-conserving nuclear momentum rescale.


def _total_energy_conserved(vel, vel_new, mass, delta_energy):
    # kinetic change must cancel the electronic energy change
    return kinetic_energy(vel_new, mass) - kinetic_energy(vel, mass) + delta_energy


# ---- uniform (1-D unambiguous) rescale -------------------------------------

def test_uniform_rescale_conserves_energy():
    vel = np.array([[0.5, -0.3, 0.2], [0.1, 0.15, -0.4]])
    mass = np.array([[1000.0] * 3, [2000.0] * 3])
    delta = 0.003  # electronic energy went up -> kinetic must drop
    vel_new, frustrated = rescale_velocity(vel, mass, delta)
    assert not frustrated
    assert _total_energy_conserved(vel, vel_new, mass, delta) == pytest.approx(0.0, abs=1e-12)
    # velocity direction preserved (uniform scale), magnitude reduced
    scale = np.sqrt((kinetic_energy(vel, mass) - delta) / kinetic_energy(vel, mass))
    np.testing.assert_allclose(vel_new, scale * vel)
    assert scale < 1.0


def test_uniform_rescale_releases_energy():
    vel = np.array([1.0, -2.0])
    mass = np.array([1.0, 1.0])
    delta = -0.5  # electronic energy dropped -> kinetic increases
    vel_new, frustrated = rescale_velocity(vel, mass, delta)
    assert not frustrated
    assert kinetic_energy(vel_new, mass) > kinetic_energy(vel, mass)
    assert _total_energy_conserved(vel, vel_new, mass, delta) == pytest.approx(0.0, abs=1e-12)


def test_uniform_rescale_frustrated_when_insufficient_ke():
    vel = np.array([0.1, 0.1])
    mass = np.array([1.0, 1.0])
    ke = kinetic_energy(vel, mass)
    vel_new, frustrated = rescale_velocity(vel, mass, delta_energy=ke + 1.0)
    assert frustrated
    np.testing.assert_array_equal(vel_new, vel)  # unchanged


def test_uniform_zero_delta_is_noop():
    vel = np.array([0.3, -0.1])
    mass = np.array([1.0, 2.0])
    vel_new, frustrated = rescale_velocity(vel, mass, 0.0)
    assert not frustrated
    np.testing.assert_allclose(vel_new, vel)


# ---- directional (NAC-direction) rescale -----------------------------------

def test_directional_rescale_conserves_energy():
    vel = np.array([0.4, -0.2, 0.6])
    mass = np.array([1000.0, 1500.0, 2000.0])
    direction = np.array([1.0, -1.0, 0.5])
    delta = 0.002
    vel_new, frustrated = rescale_velocity(vel, mass, delta, direction=direction)
    assert not frustrated
    assert _total_energy_conserved(vel, vel_new, mass, delta) == pytest.approx(0.0, abs=1e-12)
    # the change lies purely along direction/mass
    change = vel_new - vel
    ratio = change / (direction / mass)
    np.testing.assert_allclose(ratio, ratio[0])


def test_directional_rescale_frustrated_when_forbidden():
    vel = np.array([0.01, 0.0])
    mass = np.array([1.0, 1.0])
    direction = np.array([1.0, 0.0])
    # demand more energy than the projected motion can supply
    vel_new, frustrated = rescale_velocity(vel, mass, delta_energy=10.0, direction=direction)
    assert frustrated
    np.testing.assert_array_equal(vel_new, vel)


def test_directional_zero_delta_is_noop():
    vel = np.array([0.5, -0.5])
    mass = np.array([1.0, 2.0])
    direction = np.array([1.0, 1.0])
    vel_new, frustrated = rescale_velocity(vel, mass, 0.0, direction=direction)
    assert not frustrated
    np.testing.assert_allclose(vel_new, vel, atol=1e-14)


def test_directional_chooses_smaller_displacement():
    vel = np.array([1.0])
    mass = np.array([1.0])
    direction = np.array([1.0])
    vel_new, _ = rescale_velocity(vel, mass, delta_energy=0.1, direction=direction)
    # smaller-|gamma| root keeps the velocity closest to the original
    assert abs(vel_new[0] - vel[0]) < abs(vel[0])


# ---- 1-D equivalence of the two schemes (paper claim) ----------------------

def test_uniform_and_directional_equivalent_in_1d():
    # Papers: in 1-D, scaling the whole vector and rescaling along the NAC
    # direction are equivalent. Compare resulting speeds.
    vel = np.array([0.7])
    mass = np.array([1200.0])
    direction = np.array([3.14])  # any nonzero 1-D direction
    delta = 0.0009
    v_uniform, f1 = rescale_velocity(vel, mass, delta)
    v_dir, f2 = rescale_velocity(vel, mass, delta, direction=direction)
    assert not f1 and not f2
    assert abs(v_uniform[0]) == pytest.approx(abs(v_dir[0]))
    assert kinetic_energy(v_uniform, mass) == pytest.approx(kinetic_energy(v_dir, mass))
