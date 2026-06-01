import matplotlib.pyplot as plt
from tides.parse_rt import parse_output
from tides.rt_spec import abs_spec

result = parse_output('Water_DHF_UV-Vis.pyo')
w, osc_str = abs_spec(result['time'], result['dipole'], 0.0001)

plt.figure()
plt.plot(w*27.2114, osc_str[:,0], label='x')
plt.plot(w*27.2114, osc_str[:,1], label='y')
plt.plot(w*27.2114, osc_str[:,2], label='z')
plt.xlabel('Frequency (eV)')
plt.xlim([0,25])
plt.ylabel('Intensity (arb. units)')
plt.legend()
plt.savefig('Water_DHF_UV-Vis_Spectrum.png', bbox_inches='tight')

plt.figure()
plt.plot(result['time'], result['energy']*27.2114)
plt.ylabel('Energy (eV)')
plt.xlabel('Time (au)')
plt.savefig('Water_DHF_UV-Vis_Energy.png', bbox_inches='tight')
