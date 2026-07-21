import numpy as np
from pyscf import gto, dft
from tides.rt_scf import RT_SCF
from tides.rt_cap import MOCAP
from tides.rt_vapp import ElectricField

'''
Original calculation: https://nwchemgit.github.io/RT-TDDFT.html#mo-cap-example
'''

# Prepare SCF object
mol = gto.M(
	verbose = 0,
	atom='''
  O     0.00000043     0.11188833     0.00000000
  H     0.76000350    -0.47275229     0.00000000
  H    -0.76000393    -0.47275063     0.00000000
  ''',
	basis='6-31G*',
    spin = 0)

rks = dft.RKS(mol)
rks.xc = 'PBE0'
rks.kernel()

rt_scf = RT_SCF(rks, 0.2, 250)
rt_scf.observables.update(dipole=True)

# Define delta field
delta_field = ElectricField('delta', [0.0, 0.0, 0.0001])

# Define MOCAP
CAP = MOCAP(1.0, 0.5, 1.0, 100.0)

# Add both fields
rt_scf.add_potential(CAP, delta_field)

# Run dynamics
rt_scf.kernel()

