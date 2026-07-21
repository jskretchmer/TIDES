import numpy as np
from pyscf import gto, scf
from tides import rt_scf


'''
Here I'll define a simple sin-wave electric field from the input file.
'''

# Build mol

h2o_mol = gto.M(atom='''
  O     0.00000000    -0.00001441    -0.34824012
  H    -0.00000000     0.76001092    -0.93285191
  H     0.00000000    -0.75999650    -0.93290797
        ''', basis='6-31G')

# Run RHF for water
h2o = scf.RHF(h2o_mol)
h2o.kernel()

# Create RT_SCF object
rt_h2o = rt_scf.RT_SCF(h2o, 0.2, 10, filename=None, prop='magnus_interpol', frequency=1, orth=None, chkfile=None, verbose=3)

# Define custom field class. Make sure the calculate_potential method is written correctly and return matrix of correct shape (IN THE NON-ORTHOGONAL AO BASIS)
class SinWaveEField:
    def __init__(self, amplitude, frequency, phase):
        # Store field parameters as class attributes upon initialization
        self.amplitude = np.array(amplitude)
        self.frequency = frequency
        self.phase = phase
        
    def calculate_potential(self, rt_scf):
        # This is the function that get's called by rt_scf at every time step.

        # To calculate the electric field coupling, we will need the Efield at the current time. In our case, it's just a simple sin wave.
        Efield = self.amplitude * np.sin(self.frequency * rt_scf.current_time + self.phase)

        # Next, we'll need the transition dipole integral in the AO basis. This can be accessed using PySCF's mol.intor module.
        tdip = -1 * rt_scf._scf.mol.intor('int1e_r', comp=3)

        # tdip is a 3xNxN matrix (first index is x,y,z)
        # Now calculate the potential term, summing the terms for the x, y, and z coupling.
        vapp = -1 * np.einsum('xij,x->ij', tdip, Efield)

        # Finally, return this matrix to be added to the Fock matrix
        return vapp


# Create instance of sin-wave field, choosing arbitrary parameters
sinwave_instance = SinWaveEField(amplitude=[0,0,0.0001], frequency=0.25, phase=0)

# Add field to rt_h2o
rt_h2o.add_potential(sinwave_instance)


rt_h2o.kernel()
