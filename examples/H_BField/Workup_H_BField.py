import matplotlib.pyplot as plt
from tides.parse_rt import parse_output

result = parse_output('H_BField.out')


plt.figure()
plt.plot(result['time'] / 41340, result['mag'][:,0], label='x')
plt.plot(result['time'] / 41340, result['mag'][:,1], label='y')
plt.plot(result['time'] / 41340, result['mag'][:,2], label='z')
plt.ylabel('Magnetization (au)')
plt.xlabel('Time (ps)')
plt.legend()
plt.savefig('H_BField_Mag.png', bbox_inches='tight')

