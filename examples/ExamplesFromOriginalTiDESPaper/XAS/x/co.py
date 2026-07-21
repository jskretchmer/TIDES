import numpy as np
from pyscf import gto, scf, dft
from tides import RT_SCF
from tides.analysis.rt_spec import abs_spec
from tides import ElectricField
from sapporo import sapporo_c, sapporo_o

mol = gto.M(
	verbose = 0,
	atom='''
C           0.00000        0.00000        0.000
O           0.00000        0.00000        1.128
''',
basis={'C': sapporo_c, 'O': sapporo_o},
    spin = 0)

co = dft.RKS(mol)
co.xc = 'B3LYP'
co.kernel()

rt_co = RT_SCF(co, 0.02, 2000)
rt_co.observables.update(dipole=True)

delta_field = ElectricField('delta', [0.01, 0.0000, 0.0000])

rt_co.add_potential(delta_field)

rt_co.kernel()

