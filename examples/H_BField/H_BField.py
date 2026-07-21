from pyscf import gto, scf, dft
from tides import rt_scf, rt_vapp
from tides.staticfield import static_bfield

'''
Hydrogen Atom in a static B-Field
Recreated from https://doi.org/10.1063/1.4902884
'''


mag_z = 0.000085 # in au

# Build mol
mol = gto.M(
	verbose = 0,
	atom='H 0 0 0',
	basis='STO-3G',
    spin = 1)

# Build and run GHF object
mf = scf.ghf.GHF(mol)
mf.kernel()

# Add BField (this overwrites the hcore)
static_bfield(mf, [0,0,mag_z])

# Create RT_SCF object
rt_mf = rt_scf.RT_SCF(mf, 0.1, 102500)

# Specify propagator
rt_mf.prop = 'magnus_step' 
# In virtually all cases, the default 'magnus_interpol' is a more robust integrator. 
# However the magnus_step (or MMUT) integrator is what was used in the original paper

# Declare observables, in our case we only care about the magnetization
rt_mf.observables.update(mag=True)

# Start propagation
rt_mf.kernel()
