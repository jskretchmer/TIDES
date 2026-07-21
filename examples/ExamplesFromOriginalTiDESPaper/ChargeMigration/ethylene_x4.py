import numpy as np
from pyscf import gto, scf, dft
from tides import rt_scf, rt_utils, basis_utils


e1 = gto.M(
    atom='''
C           0.00010        0.00000       -0.00266
C           0.00010        0.00000        1.33266
H           0.92879        0.00000       -0.57584
H          -0.92884        0.00000       -0.57547
H           0.92879        0.00000        1.90584
H          -0.92884        0.00000        1.90547
    ''',
    basis='6-31G*', spin=1, charge=1)


e2 = gto.M(
    atom='''
C           0.00010        3.00000       -0.00266
C           0.00010        3.00000        1.33266
H           0.92879        3.00000       -0.57584
H          -0.92884        3.00000       -0.57547
H           0.92879        3.00000        1.90584
H          -0.92884        3.00000        1.90547
    ''',
    basis='6-31G*', spin=0)


e3 = gto.M(
    atom='''
C           0.00010        6.00000       -0.00266
C           0.00010        6.00000        1.33266
H           0.92879        6.00000       -0.57584
H          -0.92884        6.00000       -0.57547
H           0.92879        6.00000        1.90584
H          -0.92884        6.00000        1.90547
    ''',
    basis='6-31G*', spin=0)


e4 = gto.M(
    atom='''
C           0.00010        9.00000       -0.00266
C           0.00010        9.00000        1.33266
H           0.92879        9.00000       -0.57584
H          -0.92884        9.00000       -0.57547
H           0.92879        9.00000        1.90584
H          -0.92884        9.00000        1.90547
    ''',
    basis='6-31G*', spin=0)

mol = gto.M(verbose=0,
    atom='''
C           0.00010        0.00000       -0.00266
C           0.00010        0.00000        1.33266
H           0.92879        0.00000       -0.57584
H          -0.92884        0.00000       -0.57547
H           0.92879        0.00000        1.90584
H          -0.92884        0.00000        1.90547

C           0.00010        3.00000       -0.00266
C           0.00010        3.00000        1.33266
H           0.92879        3.00000       -0.57584
H          -0.92884        3.00000       -0.57547
H           0.92879        3.00000        1.90584
H          -0.92884        3.00000        1.90547

C           0.00010        6.00000       -0.00266
C           0.00010        6.00000        1.33266
H           0.92879        6.00000       -0.57584
H          -0.92884        6.00000       -0.57547
H           0.92879        6.00000        1.90584
H          -0.92884        6.00000        1.90547

C           0.00010        9.00000       -0.00266
C           0.00010        9.00000        1.33266
H           0.92879        9.00000       -0.57584
H          -0.92884        9.00000       -0.57547
H           0.92879        9.00000        1.90584
H          -0.92884        9.00000        1.90547
    ''',
    basis='6-31G*',
    spin = 1, charge=1)

e1 = dft.UKS(e1)
e1.xc = 'CAMB3LYP'
e1.kernel()

e2 = dft.UKS(e2)
e2.xc = 'CAMB3LYP'
e2.kernel()

e3 = dft.UKS(e3)
e3.xc = 'CAMB3LYP'
e3.kernel()

e4 = dft.UKS(e4)
e4.xc = 'CAMB3LYP'
e4.kernel()

ethylene_x4 = dft.UKS(mol)
ethylene_x4.xc = 'CAMB3LYP'
ethylene_x4.kernel()


noscf_orbitals = basis_utils.noscfbasis(ethylene_x4,e1,e2,e3,e4)
ethylene_x4.mo_coeff = noscf_orbitals
np.savetxt('NOSCF_ORBITALS_ALPHA.txt', noscf_orbitals[0])
np.savetxt('NOSCF_ORBITALS_BETA.txt', noscf_orbitals[1])

rt_ethylene_x4 = rt_scf.RT_SCF(ethylene_x4, 0.5, 500)
rt_ethylene_x4.observables.update(energy=True, hirsh_charge=True, mulliken_atom_charge=True, mo_occ_separate=True)
rt_utils.input_fragments(rt_ethylene_x4, e1, e2, e3, e4)

rt_ethylene_x4.kernel()

