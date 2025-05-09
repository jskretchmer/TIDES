"""
See https://doi.org/10.1063/1.3575587
'Magnetic circular dichroism in real-time time-dependent density functional theory'
"""

from pyscf import gto, scf
import numpy as np
from tides import rt_scf
from tides.rt_vapp import ElectricField

end_time = 500
dt = .25

mol = gto.M(
    verbose = 0,
    atom='''
O	0.8083510	-0.7692740	-0.2534320
C	-1.5023210	0.1070440	-0.1441120
H	-1.4118320	0.3428380	-1.2049650
H	-2.0725070	-0.8192750	-0.0474450
H	-2.0632360	0.9048160	0.3483070
C	-0.1447150	-0.0473440	0.4826890
H	-0.1471290	-0.2893880	1.5420300
C	1.0445900	0.6001070	-0.0512630
H	0.9661610	1.2204190	-0.9378300
H	1.8764100	0.8359330	0.6034700
  ''',
    basis= 'ccpvdz',
    unit = 'A',
    spin = 0,
)

# This part isn't necessary, but I like to see the progress
from tides import rt_output
def _custom_dipole(rt_scf):
    den_ao = rt_scf.den_ao
    mag_mom = rt_scf._scf.mol.intor('int1e_giao_irjxp') * 137.036 / 2
    # mag_mom = np.tile(mag_mom, (1,2,2))
    dipole = np.imag(np.einsum("ij,xji->x",den_ao, mag_mom))
    # dipole = rt_scf._dipole
    rt_scf._log.note(f'Total Dipole Moment [X, Y, Z] (AU): {" ".join(map(str,dipole))} \n')
    c_time = rt_scf.current_time
    print(f"{fn}:  t = {c_time:.1f}   {int(100*c_time/end_time)}%")

# hijack the dipole printer :)
rt_output._print_dipole = _custom_dipole

# Electric field amplitudes
# The key refers to the direction to be measured
fields = {
    "x_pos": [0.0001, 0, 0],
    "y_pos": [0, 0.0001, 0],
    "z_pos": [0, 0, 0.0001]
}

for fn, E_amp in fields.items():
    _scf = scf.ghf.GHF(mol)
    _scf.kernel()

    rt_mf = rt_scf.RT_SCF(_scf, dt, end_time, filename=fn)
    rt_mf.observables.update(mag=True)
    rt_mf.add_potential(ElectricField('delta', E_amp))
    rt_mf.kernel()

