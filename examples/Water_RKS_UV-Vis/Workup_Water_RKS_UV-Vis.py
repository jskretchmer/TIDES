import matplotlib.pyplot as plt
from tides.parse_rt import parse_output
from tides.rt_spec import abs_spec

result = parse_output('Water_RKS_UV-Vis.out')
w, osc_str = abs_spec(result['time'], result['dipole'], 0.0001, pad=50000, damp=50)

plt.figure()
plt.plot(w*27.2114, osc_str.sum(axis=1), 'k')
plt.xlabel('Frequency (eV)')
plt.xlim([0,25])
plt.ylabel('Intensity (arb. units)')
plt.savefig('Water_RKS_UV-Vis_Spectrum.png', bbox_inches='tight')

fig, axs = plt.subplots(1,3, figsize=(6.4*3, 4.8))
plt.subplots_adjust(wspace=0.3)
axs[0].plot(result['time'], result['dipole'][:,0])
axs[0].set_ylabel('x Dipole (au)')
axs[0].set_xlabel('Time (au)')

axs[1].plot(result['time'], result['dipole'][:,1])
axs[1].set_ylabel('x Dipole (au)')
axs[1].set_xlabel('Time (au)')

axs[2].plot(result['time'], result['dipole'][:,2])
axs[2].set_ylabel('z Dipole (au)')
axs[2].set_xlabel('Time (au)')

plt.savefig('Water_RKS_UV-Vis_Dipole.png', bbox_inches='tight')

