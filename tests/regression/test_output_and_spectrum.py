import numpy as np
from tides import RT_SCF, ElectricField
from tides.analysis import parse_rt
from tides.analysis.rt_spec import abs_spec
from rt_helpers import water, NULL


def test_each_observable_is_printed_once_per_time(tmp_path):
    out = str(tmp_path / 'run.out')
    rt = RT_SCF(water(), 0.2, 2.0, filename=out, prop='etrs', chkfile=None)
    rt.observables.update(dipole=True, energy=True, charge=True)
    rt.add_potential(ElectricField('delta', [0.0, 0.0, 1e-4]))
    rt.kernel()
    rt._fh.close()
    data = parse_rt.parse_output(out)
    assert len(data['time']) == 11
    assert len(data['dipole']) == len(data['energy']) == len(data['charge']) == len(data['time'])
    # so the parsed arrays can go straight into the spectrum routine, damping included
    w, s = abs_spec(data['time'], data['dipole'], kick_str=1e-4, damp=50)
    assert s.shape == (len(w), 3)


def test_abs_spec_frequency_axis_matches_fft_bins():
    n, dt, k = 1000, 0.2, 37
    t = dt * np.arange(n)
    w0 = 2 * np.pi * k / (n * dt)                      # exactly on FFT bin k
    dip = np.zeros((n, 3))
    dip[:, 2] = np.sin(w0 * t)
    w, s = abs_spec(t, dip, kick_str=1.0, preprocess_zero=False)
    assert np.argmax(s[:, 2]) == k
    assert abs(w[k] - w0) < 1e-12
    # a segment that does not start at t = 0 (e.g. after a restart) has the same axis
    w_shift, _ = abs_spec(t + 123.4, dip, kick_str=1.0, preprocess_zero=False)
    assert np.allclose(w_shift, w)


def test_delta_kick_fires_when_centre_is_not_a_clean_float():
    def final_dipole(center):
        mf = water()
        rt = RT_SCF(mf, 0.2, 2.0, filename=NULL, prop='etrs', chkfile=None)
        if center is not None:
            rt.add_potential(ElectricField('delta', [0.0, 0.0, 1e-3], center=center))
        rt.kernel()
        return mf.dip_moment(dm=rt.den_ao, unit='A.U.', verbose=0)[2]

    mu0 = final_dipole(None)
    assert abs(final_dipole(0.0) - mu0) > 1e-6
    assert abs(final_dipole(0.6) - mu0) > 1e-6        # 0.2 + 0.2 + 0.2 == 0.6000000000000001
