import matplotlib.pyplot as plt
from tides.parse_rt import parse_output

result = parse_output('Li_ChargeTransfer.out')


plt.figure()
plt.plot(result['time'], 3 - result['mulliken_atom_charge'][:,0], label='Li1')
plt.plot(result['time'], 3 - result['mulliken_atom_charge'][:,1], label='Li2')
plt.ylabel('Charge')
plt.xlabel('Time (au)')
plt.legend()
plt.savefig('Li_ChargeTransfer_Mulliken.png', bbox_inches='tight')

plt.figure()
plt.plot(result['time'], 3 - result['hirsh_atom_charge'][:,0], label='Li1')
plt.plot(result['time'], 3 - result['hirsh_atom_charge'][:,1], label='Li2')
plt.ylabel('Charge')
plt.xlabel('Time (au)')
plt.legend()
plt.savefig('Li_ChargeTransfer_Hirsh.png', bbox_inches='tight')
