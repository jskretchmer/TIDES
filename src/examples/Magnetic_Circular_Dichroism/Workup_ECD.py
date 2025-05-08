"""
Computes integrals
"""

import numpy as np
from scipy.integrate import simpson
from tides.parse_rt import parse_output
from tides.rt_spec import abs_spec

# Field parameters
dt = .25
total_time = 500
E_mag = 0.0002

dE = 0.001
E_bins = np.arange(0,10/27, dE)
S_func = np.zeros(E_bins.shape)

# for i, c in enumerate(['x']):
for i, c in enumerate(['x', 'y', 'z']):
    try:
        p_pos = parse_output(f'{c}_pos.txt')
    except FileNotFoundError:
        break

    t = p_pos['time']
    dip  = np.real(p_pos['mag'][:,i])
    assert t.shape[0] == total_time//dt + 1

    # Remove t=0 offset and apply constants
    dip = (dip - dip[0])/( np.pi * E_mag )
    decay = 0.008
    falloff = np.exp(-t**2 * decay**2 / 2)
    print("Decay at last timestep", falloff[-1])
    dip *= falloff

    pad = 1000
    t = np.arange(-pad * dt, total_time + pad*dt + dt, dt)
    dip = np.pad(dip, (pad,pad), constant_values=(0,0))
    assert t.shape[0] == dip.shape[0]

    for e in range(E_bins.shape[0]):
        S_func[e] += simpson(dip * np.cos(E_bins[e] * t), dx=dt)

print("Computed S_func")

import matplotlib.pyplot as plt
plt.figure(dpi=109, figsize=(19,11))
plt.rcParams.update({'lines.linewidth':4,'font.size':16})
# plt.plot(t, dip)
plt.plot(E_bins*27, S_func)
# plt.ylim(-80,80)
plt.xlabel('Energy (eV)')
plt.ylabel('Dipole response')
plt.tight_layout()
plt.savefig('MCD.svg', bbox_inches='tight')

