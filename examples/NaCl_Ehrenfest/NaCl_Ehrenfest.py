from pyscf import gto, dft, scf
import numpy as np
from tides import rt_ehrenfest

'''
Recreated from:
https://doi.org/10.1063/1.2008258
'''

nacl = gto.Mole(basis = '321g')

nacl.atom = '''
 Na     0.00000000     0.00000000     0.00000000
 Cl     0.00000000     2.42100000     0.00000000
'''

nacl.build()

nacl = scf.UHF(nacl)

nacl.kernel()

ehrenfest_nacl = rt_ehrenfest.RT_Ehrenfest(nacl, 0.25, 12300, verbose=3, Ne_step=10, N_step=10)
ehrenfest_nacl.nuc.vel[0,1] = 1.1038e-3
ehrenfest_nacl.nuc.vel[1,1] = -0.7254e-3
ehrenfest_nacl.observables.update(nuclei=True, energy=True)
ehrenfest_nacl.kernel()

