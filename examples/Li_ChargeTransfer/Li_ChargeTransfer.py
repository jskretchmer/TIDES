from pyscf import gto, scf, dft
from tides import rt_scf
from tides import rt_utils
from tides import basis_utils

dimer = gto.Mole()
Li1 = gto.Mole()
Li2 = gto.Mole()

dimer.atom = '''
Li 0.0 0.0 0.0
Li 0.0 0.0 5.0
'''

Li1.atom = '''
Li 0.0 0.0 0.0
'''

Li2.atom = '''
Li 0.0 0.0 5.0
'''

dimer.basis = '6-31G*'
Li1.basis = '6-31G*'
Li2.basis = '6-31G*'

dimer.charge = +1
dimer.spin = 1
dimer.build()

Li1.charge = +1
Li1.spin = 0
Li2.charge = 0
Li2.spin = 1
Li1.build()
Li2.build()

dimer = scf.UHF(dimer)
Li1 = scf.UHF(Li1)
Li2 = scf.UHF(Li2)


dimer.kernel()
Li1.kernel()
Li2.kernel()

dimer.mo_coeff = basis_utils.noscfbasis(dimer,Li1,Li2)
rt_mf = rt_scf.RT_SCF(dimer,0.05, 500)
rt_mf.prop = 'rk4' # No good reason to use rk4, but it works with small timesteps.
rt_mf.observables.update(mulliken_atom_charge=True, hirsh_atom_charge=True)

rt_mf.kernel()
