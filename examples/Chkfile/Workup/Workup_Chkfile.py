import numpy as np
import matplotlib.pyplot as plt
from tides.parse_rt import parse_output

result_p1 = parse_output('Part1.out')
result_p2 = parse_output('Part2.out')

time = np.concatenate((result_p1['time'], result_p2['time']))
mulliken = np.concatenate((result_p1['mulliken_atom_charge'], result_p2['mulliken_atom_charge']), axis=0)
hirsh = np.concatenate((result_p1['hirsh_atom_charge'], result_p2['hirsh_atom_charge']), axis=0)

plt.figure()
plt.plot(time, 3 - mulliken[:,0], label='Li1')
plt.plot(time, 3 - mulliken[:,1], label='Li2')
plt.ylabel('Charge')
plt.xlabel('Time (au)')
plt.legend()
plt.savefig('Chkfile_Li_ChargeTransfer_Mulliken.png', bbox_inches='tight')

plt.figure()
plt.plot(time, 3 - hirsh[:,0], label='Li1')
plt.plot(time, 3 - hirsh[:,1], label='Li2')
plt.ylabel('Charge')
plt.xlabel('Time (au)')
plt.legend()
plt.savefig('Chkfile_Li_ChargeTransfer_Hirsh.png', bbox_inches='tight')
