from pyscf import gto, scf
import numpy as np
from tides import rt_ehrenfest

# Build mol
mol = gto.M(atom='''
  H    0.0 0.0 0.0
  H    0.0 0.0 0.75
        ''', basis='6-31G')

# Build UHF object
uhf = scf.UHF(mol)

# Run SCF
uhf.kernel()

# Declare propagation parameters
rt_ehrenfest = rt_ehrenfest.RT_Ehrenfest(uhf, 0.05, 500, 
        Ne_step=1, N_step=1)

# Declare observables
rt_ehrenfest.observables.update(energy=True, dipole=True, nuclei=True)

# Let's start the simulation with 10eV of vibrational energy within the H-H bond.
# KE = \sum_i{0.5 m_i v_i**2}
# v_i = \sqrt{2KE_i/m_i}
# KE_i = 10 / 2
# 1 au = 27.2114 eV
# H mass = 1836 m_e

KE_i = 5.0 # 5.0 eV for each H, giving total KE of 10.0 eV
init_velo = np.sqrt(2*(KE_i/27.2114)/1836)

# Set velocities in the z direction.
rt_ehrenfest.nuc.vel[0,2] = -1 * init_velo # Make sure velocities are in opposite directions
rt_ehrenfest.nuc.vel[1,2] = init_velo 

# Start propagation
rt_ehrenfest.kernel()


