from pyscf import gto, dft
from tides import rt_scf

"""
See 15-nlc_functionals.py in PySCF's examples/dft folder for more details on how to use non-local correlation functionals.
"""
# Build mol
mol = gto.M(atom='''
  O     0.00000000    -0.00001441    -0.34824012
  H    -0.00000000     0.76001092    -0.93285191
  H     0.00000000    -0.75999650    -0.93290797
        ''', basis='6-31G')
    
# Build RKS object
rks = dft.RKS(mol)

# Use non-local dispersion functional
rks.xc = 'wb97m_v'

# Run SCF
rks.kernel()

# Declare propagation parameters
rt_scf = rt_scf.RT_SCF(rks, 
timestep=1.0, max_time=100)
    
# Declare observables
rt_scf.observables.update(energy=True)

# Start propagation
rt_scf.kernel()
