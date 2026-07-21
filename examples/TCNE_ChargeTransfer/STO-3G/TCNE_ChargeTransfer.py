from pyscf import gto, dft, scf
import numpy as np
from tides import rt_scf, rt_utils, basis_utils


'''
Recreated from:
https://nwchemgit.github.io/RT-TDDFT.html#charge-transfer-between-a-tcne-dimer

We use STO-3G instead of a charge density fitting basis
'''


# First build and run SCF for the TCNE dimer as well as the top/bottom TCNE monomers
dimer = gto.Mole()
top = gto.Mole()
bottom = gto.Mole()

dimer.atom = '''
 C    -1.77576486     0.66496556     0.00004199
 N    -2.94676621     0.71379797     0.00004388
 C    -0.36046718     0.62491168     0.00003506
 C     0.36049301    -0.62492429    -0.00004895
 C     1.77579907    -0.66504145    -0.00006082
 N     2.94680364    -0.71382258    -0.00006592
 C    -0.31262746    -1.87038951    -0.00011201
 N    -0.85519492    -2.90926164    -0.00016331
 C     0.31276207     1.87031662     0.00010870
 N     0.85498782     2.90938919     0.00016857
#---
 C    -1.77576486     0.66496556     3.00004199
 N    -2.94676621     0.71379797     3.00004388
 C    -0.36046718     0.62491168     3.00003506
 C     0.36049301    -0.62492429     2.99995105
 C     1.77579907    -0.66504145     2.99993918
 N     2.94680364    -0.71382258     2.99993408
 C    -0.31262746    -1.87038951     2.99988799
 N    -0.85519492    -2.90926164     2.99983669
 C     0.31276207     1.87031662     3.00010870
 N     0.85498782     2.90938919     3.00016857
'''

top.atom = '''
 C    -1.77576486     0.66496556     3.00004199
 N    -2.94676621     0.71379797     3.00004388
 C    -0.36046718     0.62491168     3.00003506
 C     0.36049301    -0.62492429     2.99995105
 C     1.77579907    -0.66504145     2.99993918
 N     2.94680364    -0.71382258     2.99993408
 C    -0.31262746    -1.87038951     2.99988799
 N    -0.85519492    -2.90926164     2.99983669
 C     0.31276207     1.87031662     3.00010870
 N     0.85498782     2.90938919     3.00016857
'''


bottom.atom = '''
 C    -1.77576486     0.66496556     0.00004199
 N    -2.94676621     0.71379797     0.00004388
 C    -0.36046718     0.62491168     0.00003506
 C     0.36049301    -0.62492429    -0.00004895
 C     1.77579907    -0.66504145    -0.00006082
 N     2.94680364    -0.71382258    -0.00006592
 C    -0.31262746    -1.87038951    -0.00011201
 N    -0.85519492    -2.90926164    -0.00016331
 C     0.31276207     1.87031662     0.00010870
 N     0.85498782     2.90938919     0.00016857
'''

dimer.basis = 'STO-3G'
top.basis = 'STO-3G'
bottom.basis = 'STO-3G'

dimer.charge = -1
dimer.spin = 1
dimer.build()

top.charge = 0
top.spin = 0
bottom.charge = -1
bottom.spin = 1
top.build()
bottom.build()

dimer = scf.UKS(dimer)
top = scf.UKS(top)
bottom = scf.UKS(bottom)

dimer.xc = 'CAMB3LYP'
top.xc = 'CAMB3LYP'
bottom.xc = 'CAMB3LYP'

dimer.kernel()
top.kernel()
bottom.kernel()


# Now overwrite the dimer orbitals with the SCF orbitals of the bottom/top monomers
dimer.mo_coeff = basis_utils.noscfbasis(dimer,bottom,top)
rt_scf = rt_scf.RT_SCF(dimer, 0.2, 500)
rt_scf.observables.update(charge=True)

# We'll use the input_fragments function here so that the Mulliken charges on the monomers are calculated
rt_utils.input_fragments(rt_scf,bottom,top)

# Run dynamics
rt_scf.kernel()
