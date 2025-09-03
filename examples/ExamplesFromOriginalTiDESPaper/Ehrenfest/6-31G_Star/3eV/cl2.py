from pyscf import gto, dft
from tides import rt_ehrenfest, rt_utils
import numpy as np

# Build Cl2 molecule
mol = gto.M(
    verbose = 0,
    atom='''
Cl           0.00000000 0.00000000 0.00000000
Cl           0.00000000 0.00000000 2.00000000
''',
    basis='6-31G*', spin=0)

# Build Unrestricted Kohn-Sham object
cl2 = dft.UKS(mol)
cl2.xc = 'B3LYP'
cl2.kernel()

# Create RT_Ehrenfest object
rt_cl2 = rt_ehrenfest.RT_Ehrenfest(cl2, 0.5, 5000, 
        frequency=100, Ne_step=1, N_step=1)

# Set initial velocity
init_eV = 3.0
init_velo = np.sqrt((init_eV/(2*27.2114))*2/64260)
rt_cl2.nuc.vel[0,2] = -1 * init_velo
rt_cl2.nuc.vel[1,2] = init_velo

# Declare which observables to be calculated/printed
rt_cl2.observables.update(energy=True, nuclei=True)

rt_cl2.kernel()
