from pyscf import gto, scf
from tides import rt_scf
from tides.rt_vapp import ElectricField

# Build mol
mol = gto.M(atom='''
  O     0.00000000    -0.00001441    -0.34824012
  H    -0.00000000     0.76001092    -0.93285191
  H     0.00000000    -0.75999650    -0.93290797
        ''', basis='6-31G')
    
# Build RHF object
rhf = scf.RHF(mol)

# Run SCF
rhf.kernel()

# Declare propagation parameters
rt_scf = rt_scf.RT_SCF(rhf, 
timestep=1.0, max_time=1000)
    
# Declare observables
rt_scf.observables.update(energy=True, dipole=True)

# Add electric field
delta_field = ElectricField('delta', [0.0001, 0.0001, 0.0001]) 
rt_scf.add_potential(delta_field)
# It would be better to perform 3 separate simulations for fields polarized in each direction, 
# but it makes a negligible difference here

# Start propagation
rt_scf.kernel()
