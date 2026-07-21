import numpy as np
import matplotlib.pyplot as plt
from tides.parse_rt import parse_output

result = parse_output('Water_ResonantExcitation.out')

efield = 0.0001 * np.exp(-(result['time']-393.3)**2/(2*64.8**2)) * np.sin(0.3768*result['time'])

fig, ax = plt.subplots(2, figsize=(8,10))
ax[0].plot(result['time'] / 41.34, efield, 'r', linewidth=1)
ax[0].set_xlabel('Time (fs)', fontsize=16)
ax[0].set_ylabel('z Electric Field (au)', fontsize=16)
ax[0].ticklabel_format(style='plain')

ax[1].plot(result['time'] / 41.34, result['dipole'][:,2] / 41.34, 'k', linewidth=1)
ax[1].set_xlabel('Time (fs)', fontsize=16)
ax[1].set_ylabel('z Dipole (au)', fontsize=16)
ax[1].ticklabel_format(style='plain')
plt.savefig("Water_ResonantExcitation_Dipole.png", bbox_inches='tight')
