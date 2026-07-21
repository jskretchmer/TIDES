import numpy as np
from pyscf import gto, scf, dft
from tides.rt_scf import RT_SCF
from tides.rt_vapp import ElectricField

'''
Original calculation: https://nwchemgit.github.io/RT-TDDFT.html#resonant-ultraviolet-excitation-of-water
'''

# SCF
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

# RT_SCF
rt_scf = RT_SCF(rks, 0.2, 1000)
rt_scf.observables.update(energy=True, dipole=True)

# Gaussian Field
gaussian_field = ElectricField('gaussian', amplitude=[0.0, 0.0, 0.0001],
                             center=393.3, frequency=0.3768,
                              width=64.8)

rt_scf.add_potential(gaussian_field)

# -->
rt_scf.kernel()
