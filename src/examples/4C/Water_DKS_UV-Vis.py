from pyscf import gto, dft, lib
from tides import rt_scf
from tides.rt_vapp import ElectricField
import numpy as np

mol = gto.M(
    verbose = 0,
	atom='''
  O     0.00000000    -0.00001441    -0.34824012
  H    -0.00000000     0.76001092    -0.93285191
  H     0.00000000    -0.75999650    -0.93290797
  ''',
	basis='unc-sto3g',
    spin = 0,
)

dks = dft.DKS(mol)
dks.xc = 'pbe'
# options are 'col' 'ncol' 'mcol'
# for collinear, non-collinear, and multi-collinear functionals
# 'mcol' requires mcfun package
dks.collinear = 'col'
dks.kernel()

rt_mf = rt_scf.RT_SCF(dks, 0.02, 100)
rt_mf.observables.update(energy=True, dipole=True)

rt_mf.add_potential(ElectricField('delta', [0.0001, 0.0000, 0.0000]))

rt_mf.kernel()

