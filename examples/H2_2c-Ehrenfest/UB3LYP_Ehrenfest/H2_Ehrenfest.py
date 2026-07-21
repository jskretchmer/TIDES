from pyscf import gto, scf
import numpy as np
from tides import rt_ehrenfest

# Build mol
mol = gto.M(atom='''
  H    0.0 0.0 0.0
  H    0.0 0.0 0.75
        ''', basis='6-31G')

# Build UKS object
uks = scf.UKS(mol)
uks.xc = 'B3LYP'

# Run SCF (if you start with assymetric guess, this will dissociate)
#udm_guess = np.array([[[0.18568439, 0.28537676, 0.00000000, 0.00000000],
#                       [0.28537676, 0.43859311, 0.00000000, 0.00000000],
#                       [0.00000000, 0.00000000, 0.00000000, 0.00000000],
#                       [0.00000000, 0.00000000, 0.00000000, 0.00000000]],
#                      [[0.00000000, 0.00000000, 0.00000000, 0.00000000],
#                       [0.00000000, 0.00000000, 0.00000000, 0.00000000],
#                       [0.00000000, 0.00000000, 0.18568439, 0.28537676],
#                       [0.00000000, 0.00000000, 0.28537676, 0.43859311]]])
#uks.kernel(dm0 = udm_guess)
uks.kernel()

# Declare propagation parameters
rt_ehrenfest = rt_ehrenfest.RT_Ehrenfest(uks, 0.04, 4200, Ne_step=1, N_step=1,
                                         filename='H2_Ehrenfest.out', frequency=20,
                                         chkfile='H2_Ehrenfest.chk', verbose=5)

# Declare observables
rt_ehrenfest.observables.update(energy=True, nuclei=True)

# Let's start the simulation with 7.25eV of vibrational energy within the H-H bond.
# KE = \sum_i{0.5 m_i v_i**2}
# v_i = \sqrt{2KE_i/m_i}
# KE_i = 7.25 / 2
# 1 au = 27.211386246 eV
# H mass = 1836.15267343 m_e

rt_ehrenfest.nuc.mass[0] = 1836.15267343
rt_ehrenfest.nuc.mass[1] = 1836.15267343

KE_i = 3.625 # 3.625 eV for each H, giving total KE of 7.25 eV
init_velo = np.sqrt(2*(KE_i/27.211386246)/1836.15267343)

# Set velocities in the z direction.
rt_ehrenfest.nuc.vel[0,2] = -1 * init_velo # Make sure velocities are in opposite directions
rt_ehrenfest.nuc.vel[1,2] = init_velo 

# Start propagation
rt_ehrenfest.kernel()

