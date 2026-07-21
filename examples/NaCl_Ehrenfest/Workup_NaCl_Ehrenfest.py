import matplotlib.pyplot as plt
import numpy as np
from MDAnalysis.coordinates.XYZ import XYZReader
from tides.parse_rt import parse_output, get_length

result = parse_output('NaCl_Ehrenfest.out')

xyz = XYZReader('trajectory.xyz', dt=0.25*100)
xyz.units['time'] = 'au'

time = []
positions = []
for ts in xyz:
    time.append(ts.time)
    positions.append(np.array(ts.positions).astype(np.float64))
time = np.array(time)
positions = np.array(positions)

HH_dist = get_length(positions, [1,2])

plt.figure()
plt.plot(time, HH_dist)
plt.ylabel(r'R(H-H) ($\mathrm{\AA}$)')
plt.xlabel('Time (au)')
plt.savefig('NaCl_Ehrenfest_Distance.png', bbox_inches='tight')

plt.figure()
plt.plot(result['time'], (result['energy'] - result['energy'][0])*27.2114)
plt.ylabel(r'$\Delta$Energy (eV)')
plt.xlabel('Time (au)')
plt.savefig('NaCl_Ehrenfest_Energy.png', bbox_inches='tight')
