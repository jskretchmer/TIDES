import matplotlib.pyplot as plt
from tides.parse_rt import parse_output

result_without_mocap = parse_output('Water_Without_MOCAP.out')
result_mocap = parse_output('Water_MOCAP.out')

plt.figure()
ax = plt.axes()
plt.plot(result_without_mocap['time'], result_without_mocap['dipole'][:,2], 'r', linewidth=1, label='CAP off')
plt.plot(result_mocap['time'], result_mocap['dipole'][:,2], 'k', linewidth=2, label='CAP on')
plt.xlabel('Time (au)', fontsize=20)
plt.ylabel('z Dipole (au)', fontsize=20)
plt.ticklabel_format(style='plain')
plt.legend()
plt.savefig('Water_MOCAP_Dipole.png', bbox_inches='tight')
