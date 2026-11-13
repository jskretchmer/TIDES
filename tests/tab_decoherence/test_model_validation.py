import numpy as np
import pytest

from tides.decoherence import models

# Validation ladder: reproduce the Paper A model-problem behavior
# (J. Chem. Phys. 152, 234105 (2020), Sec. III, Tables/Figs). Trajectory counts are
# kept modest so the suite stays fast; the distinguishing results (exact zeros for
# parallel-state collapse) are structural and hold even with few trajectories.


def test_two_state_coherence_decays_exponentially():
    m = models.MODELS['two_state']
    tau = models.model_decoherence_times(m)[0, 1]
    times, mrho, _ = models.run_ensemble(m, ntraj=300, seed=10, record_stride=8)
    rho0 = np.sqrt(0.25 * 0.75)
    assert mrho[0, 0, 1] == pytest.approx(rho0, abs=1e-6)          # initial coherence
    # coherence tracks |rho01(0)| exp(-t/tau) within ensemble noise, and is monotone
    coh = mrho[:, 0, 1]
    assert np.all(np.diff(coh) <= 1e-6)                            # non-increasing
    # near one lifetime, within ~30% of the exponential target (300-traj noise)
    idx = int(np.argmin(np.abs(times - tau)))
    target = rho0 * np.exp(-times[idx] / tau)
    assert abs(coh[idx] - target) < 0.3 * target


def test_three_state_parallel_states_never_collapse_individually():
    # THE distinguishing result: TAB never collapses a trajectory onto one of the
    # parallel states 1, 2 (their mutual coherence never decays), so those fractions
    # are exactly zero, while ~half collapse to the non-parallel state 0.
    m = models.MODELS['three_state']
    _, _, fpops = models.run_ensemble(m, ntraj=300, seed=11, record_stride=50)
    frac = models.collapse_fractions(fpops)
    assert frac[1] == 0.0
    assert frac[2] == 0.0
    assert 0.40 < frac[0] < 0.60          # Paper A: ~0.50


def test_three_state_parallel_coherence_stays_constant():
    m = models.MODELS['three_state']
    times, mrho, _ = models.run_ensemble(m, ntraj=300, seed=12, record_stride=8)
    coh_12 = mrho[:, 1, 2]  # parallel pair: should stay ~0.25
    coh_01 = mrho[:, 0, 1]  # non-parallel: should decay to ~0
    assert np.all(np.abs(coh_12 - 0.25) < 0.04)
    assert coh_01[0] > 0.34 and coh_01[-1] < 0.05


def test_five_state_faster_decay_for_larger_force_gap():
    # Coherence between states with a larger force difference decays faster.
    m = models.MODELS['five_state']
    times, mrho, _ = models.run_ensemble(m, ntraj=150, seed=13, record_stride=10)
    idx = int(np.argmin(np.abs(times - 300.0)))
    small_gap = mrho[idx, 1, 0]   # |slope diff| = 0.010 (slowest)
    large_gap = mrho[idx, 4, 0]   # |slope diff| = 0.040 (fastest)
    assert small_gap > large_gap


def test_five_state_equal_force_gaps_decay_together():
    # The four pairs with |slope diff| = 0.010 share one decay rate.
    m = models.MODELS['five_state']
    times, mrho, _ = models.run_ensemble(m, ntraj=200, seed=14, record_stride=10)
    idx = int(np.argmin(np.abs(times - 300.0)))
    equal_gap = np.array([mrho[idx, 1, 0], mrho[idx, 2, 1],
                          mrho[idx, 3, 2], mrho[idx, 4, 3]])
    # all four within a modest spread of their common value
    assert equal_gap.std() < 0.2 * equal_gap.mean()
