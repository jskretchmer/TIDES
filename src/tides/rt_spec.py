import numpy as np
from numpy.fft import fft
from pyscf import lib


'''
Real-time SCF Spectra
'''

c = lib.param.LIGHT_SPEED
def abs_spec(time, pole, kick_str=1, pad=None, damp=None, hann_damp=None, preprocess_zero=True):
    '''
    Performs 1D Fourier Transform on time-dependent multipole moment.
    Adapted from NWChem's fft1d.m GNU Octave script (Kenneth Lopata), which can be found at https://nwchemgit.github.io/RT-TDDFT.html#absorption-spectrum-of-water
    '''

    pole_t = np.copy(pole)

    if preprocess_zero:
        pole_t -= pole_t[0,:]

    if damp:
        d = np.exp((-1 * (time - time[0])) / damp)
        pole_t *= d[:, np.newaxis]
    
    if hann_damp:
        if type(hann_damp) is not tuple and type(hann_damp) is not list:
            raise Exception('hann_damp must be a list/tuple with [t0, sigma]')
        t0 = hann_damp[0]
        sigma = hann_damp[1]
        d = np.sin(np.pi / sigma * (time - t0)) ** 2
        pole_t *= d[:, np.newaxis]
    
    if pad:
        zeros = np.zeros((pole_t.shape[1], pad))
        pole_t = np.row_stack((pole_t, zeros.T))

    n = pole_t.shape[0]
    dt = time[1] - time[0]              # Assumes constant timestep
    period = (n-1) * dt - time[0]
    dw = 2.0 * np.pi / period

    m = int(n / 2)                      # Include only positive frequencies
    wmin = 0.0
    wmax = m * dw
    w = np.linspace(wmin, wmax, m)

    osc_str = []
    for i in range(pole_t.shape[1]):
        pole_f = fft(pole_t[:,i])
        osc_str.append(np.abs(np.imag(pole_f[:m])) * (4 * np.pi) / (c * kick_str) * w)

    osc_str = np.array(osc_str).T
    return w, osc_str
