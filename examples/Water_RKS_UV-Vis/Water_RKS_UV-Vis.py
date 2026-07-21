import numpy as np
from pyscf import gto, scf, dft
from tides import rt_scf
from tides.rt_vapp import ElectricField

'''
Original calculation: https://nwchemgit.github.io/RT-TDDFT.html#absorption-spectrum-of-water
'''

# Prepare SCF
mol = gto.M(
	verbose = 0,
	atom='''
  O     0.00000000    -0.00001441    -0.34824012
  H    -0.00000000     0.76001092    -0.93285191
  H     0.00000000    -0.75999650    -0.93290797
  ''',
	basis='6-31G',
    spin = 0)

rks = dft.RKS(mol)
rks.xc = 'PBE0'
rks.kernel()

# Create RT_SCF
rt_scf = rt_scf.RT_SCF(rks, 0.2, 200)
rt_scf.observables.update(dipole=True)

# Define field
delta_field = ElectricField('delta', [0.0001, 0.0001, 0.0001]) # Applying x,y,z polarization simultaneously

# Add field
rt_scf.add_potential(delta_field)

# Go
rt_scf.kernel()
