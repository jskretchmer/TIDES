"""
See https://doi.org/10.1063/1.3575587
'Magnetic circular dichroism in real-time time-dependent density functional theory'
"""

from pyscf import gto, scf
import numpy as np
from tides import rt_scf
from tides.staticfield import static_bfield
from tides.rt_vapp import ElectricField

end_time = 1500
dt = .5

mol = gto.M(
    verbose = 0,
    atom='''
C	0.0000000	0.0000000	-1.1296200
Cl	0.0000000	0.0000000	0.6573070
H	0.0000000	1.0223540	-1.4655000
H	0.8853850	-0.5111770	-1.4655000
H	-0.8853850	-0.5111770	-1.4655000
  ''',
    basis= 'ccpvdz',
    unit = 'A',
    spin = 0,
)

# This part isn't necessary, but I like to see the progress
from tides import rt_output
def _custom_dipole(rt_scf):
    dipole = rt_scf._dipole
    rt_scf._log.note(f'Total Dipole Moment [X, Y, Z] (AU): {" ".join(map(str,dipole))} \n')
    c_time = rt_scf.current_time
    print(f"{dir}:  t = {c_time:.1f}   {int(100*c_time/end_time)}%")

# hijack the dipole printer :)
rt_output._print_dipole = _custom_dipole

# Magnetic and electric field amplitudes respectively
# The key refers to the dipole direction to be measured
# pos and neg refer to the two different components for each direction
fields = {
    "x_pos": ([0, 0, 0.001], [0, 0.0005, 0]),
    "x_neg": ([0, 0.001, 0], [0, 0, 0.0005]),
    "y_pos": ([0.001, 0, 0], [0, 0, 0.0005]),
    "y_neg": ([0, 0, 0.001], [0.0005, 0, 0]),
    "z_pos": ([0, 0.001, 0], [0.0005, 0, 0]),
    "z_neg": ([0.001, 0, 0], [0, 0.0005, 0]),
}

for dir, (B_amp, E_amp) in fields.items():
    _scf = scf.ghf.GHF(mol).x2c()
    _scf.kernel()

    static_bfield(_scf, B_amp)

    rt_mf = rt_scf.RT_SCF(_scf, dt, end_time, filename=dir)
    rt_mf.observables.update(energy=True, dipole=True)
    field = ElectricField(rt_mf, E_amp)
    rt_mf.add_potential(field)
    rt_mf.kernel()

