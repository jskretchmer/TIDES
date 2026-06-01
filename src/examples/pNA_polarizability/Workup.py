"""
Given the field strength and dipole moment, this script creates
a linear regression to solve for the polarizability via the eq.

   μ_{ij}¹(t) = α_{ij}(-ω;ω) cos(ωt)
"""

import numpy as np
from scipy.stats import linregress
import matplotlib.pyplot as plt
from tides.parse_rt import parse_output

# Field parameters
dt = .8
total_time = 750
freq = 0.0428
strength = 0.002

# Create the field
NRG = []
for i, t in enumerate(np.arange(0,total_time+dt,dt)):
    field = np.cos(freq * t)
    if (t < 2*np.pi / freq):
        field *= t * freq / (2 * np.pi)
    NRG.append(-1 * field)

# Analyze dipoles
results = []
for i, c in enumerate(['x','y','z']):
    p1 = parse_output(f'{c}_pos.txt')
    n1 = parse_output(f'{c}_neg.txt')
    t = p1['time']
    t = n1['time']
    dip_p1 = p1['dipole']
    dip_n1 = n1['dipole']

    n = min(dip_p1.shape[0], dip_n1.shape[0])
    field = np.asarray(NRG[:n])

    # Compute the susceptibility
    dip  = dip_p1[:n,i] - dip_p1[0,i]
    dip -= dip_n1[:n,i] - dip_n1[0,i]
    yi = dip/(2*strength)

    # Get α from a linear regression
    res = linregress(field, y=yi, alternative='two-sided')
    results.append(res)
    print(f"α_{c}{c} = {res.slope:.2f}±{res.stderr:.2f}")

    plt.figure()
    plt.plot(t[:n], field * res.slope, label='field')
    plt.plot(t[:n], yi, label=f"μ_{c}⁽¹⁾")
    plt.xlabel('Time')
    plt.ylabel('Linear susceptability')
    plt.legend()
    plt.savefig(f'dip_{c}_pos.png', bbox_inches='tight')

