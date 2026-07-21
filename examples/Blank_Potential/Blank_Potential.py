import numpy as np
from pyscf import gto, scf
from tides import rt_scf

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
class CUSTOM:
    def __init__(self):
        pass
        
    def calculate_potential(self, rt_scf):
        # Here I am just returning a zero matrix of the correct shape. Replace with desired form of potential.
        return np.zeros(rt_scf.fock_ao.shape)


# Create instance of custom field
custom_instance = CUSTOM()

# Add field to rt_h2o
rt_h2o.add_potential(custom_instance)


rt_h2o.kernel()
