import matplotlib.pyplot as plt
from tides.parse_rt import parse_output

result = parse_output('TCNE_ChargeTransfer.pyo')


plt.figure()
plt.plot(result['time'], result['frag_charge'][:,0], label='Bottom')
plt.plot(result['time'], result['frag_charge'][:,1], label='Top')
plt.ylabel('Charge')
plt.xlabel('Time (au)')
plt.legend()
plt.savefig('TCNE_ChargeTransfer_Charges.png', bbox_inches='tight')
