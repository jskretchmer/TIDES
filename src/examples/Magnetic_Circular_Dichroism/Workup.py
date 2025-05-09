"""
Computes integrals
"""

import numpy as np
from tides.parse_rt import parse_output
from tides.rt_spec import abs_spec

# Field parameters
dt = 0.5
total_time = 1500
E_mag = 0.0001
B_mag = 0.001 # 0.0033

dE = 0.001
E_bins = np.arange(-1,3, dE)
S_func = np.zeros(E_bins.shape)

for i, c in enumerate(['x','y','z']):
    try:
        p_pos = parse_output(f'{c}_pos.txt')
        p_neg = parse_output(f'{c}_neg.txt')
    except FileNotFoundError:
        break

    t = p_pos['time']
    dip  = p_pos['dipole']
    dip -= p_neg['dipole']

    # Remove t=0 offset and apply constants
    dip = (dip - dip[0])/(3 * np.pi * E_mag * B_mag)
    assert dip.shape[0] == t.shape[0]

    integral = np.zeros(E_bins.shape)
    func = np.outer(t, E_bins)
    S_func += (dip[:,i] @ np.cos(func)) * dt

print("Computed S_func")

x = np.arange(-1,1,dE)
gamma = 0.027
cauchy = 1/(np.pi * gamma * (1 + (x/gamma)**2))

result = np.convolve(S_func, cauchy, mode='valid')
res_E = np.arange(0,2+dE,dE)
print("DONE")

import matplotlib.pyplot as plt
plt.ion()
plt.figure(dpi=100, figsize=(10,8))
plt.plot(res_E * 219.5, -1 *result)
plt.plot(res_E[:cauchy.shape[0]] * 219.5, cauchy*4e5)
plt.plot(E_bins * 219.5, S_func * -30)
plt.xlim([0,400])
plt.gca().invert_xaxis()
plt.xlabel('Energy (k cm-1)')
plt.ylabel('Dipole response')
plt.savefig('MCD.png', bbox_inches='tight')
plt.show()

