from pyscf import gto, dft
from tides.rt_scf import RT_SCF
from tides.rt_vapp import ElectricField

# Build benzene molecule
mol = gto.M(
    verbose = 0,
    atom='''
 C            1.39442615  0.00000000  0.00000000
 C           -1.39442615  0.00000000  0.00000000
 C            0.69721307 -1.20760847  0.00000000
 C           -0.69721307  1.20760847  0.00000000
 C           -0.69721307 -1.20760847  0.00000000
 C            0.69721307  1.20760847  0.00000000
 H            2.47650069  0.00000000  0.00000000
 H           -2.47650069  0.00000000  0.00000000
 H            1.23825035 -2.14471251  0.00000000
 H           -1.23825035  2.14471251  0.00000000
 H           -1.23825035 -2.14471251  0.00000000
 H            1.23825035  2.14471251  0.00000000
''',
    basis='6-31G*', spin=0)

# Build Restricted Kohn-Sham object
benzene = dft.RKS(mol)
benzene.xc = 'B3LYP'
benzene.kernel()

# Create RT_SCF object
rt_benzene = RT_SCF(benzene, 0.2, 5000)
rt_benzene.observables['dipole'] = True

# x-polarized delta field
delta_field = ElectricField('delta', [0.0001, 0.0, 0.0])

rt_benzene.add_potential(delta_field)

rt_benzene.kernel()
