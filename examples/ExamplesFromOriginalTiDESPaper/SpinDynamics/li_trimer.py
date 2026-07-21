import numpy as np
from pyscf import gto, scf, dft
from tides.rt_scf import RT_SCF
from pyscf.scf.stability import ghf_stability ##
from tides.staticfield import static_bfield

mag_y = 0.000085 # in au

LiTrimerMol = gto.M(
	verbose = 0,
	atom='''
 Li                 0.00000000    0.0 0.909326674
 Li                 -1.05    0.0 -0.909326674
 Li                 1.05    0.0 -0.909326674
''',
	basis='3-21G',
    spin = 1)


LiTrimer = scf.GHF(LiTrimerMol)
LiTrimer.kernel()

# Run stability analysis to verify our initial state
stable = False
while not stable:
    mo_new, stable = ghf_stability(LiTrimer, return_status=True, tol=1e-8)
    dm1 = LiTrimer.make_rdm1(mo_new, LiTrimer.mo_occ)
    LiTrimer.kernel(dm1.astype(np.complex128))


# Add BField, create RT_SCF
static_bfield(LiTrimer, [0, mag_y, 0])
rt_mf = RT_SCF(LiTrimer, 0.5, 82000)

rt_mf.observables.update(mag=True, hirsh_mag=True)

# Run
rt_mf.kernel()

