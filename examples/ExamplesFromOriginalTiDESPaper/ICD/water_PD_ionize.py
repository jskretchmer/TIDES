from pyscf import gto, dft, scf
import numpy as np
from tides import rt_scf, rt_utils, basis_utils
from tides.rt_cap import MOCAP

dimer = gto.Mole()
pd = gto.Mole()
pa = gto.Mole()

dimer.atom = '''
 O1                  1.49137509   -0.00861627    0.00000100
 H1                  0.53614403    0.12578597    0.00000098
 H1                  1.86255228    0.87394752    0.00000101
 O2                  -1.41904607    0.10858435    0.00000006
 H2                  -1.76001996   -0.36619300   -0.76029109
 H2                  -1.76001996   -0.36619300    0.76029121
'''

pd.atom = '''
 O1                  1.49137509   -0.00861627    0.00000100
 H1                  0.53614403    0.12578597    0.00000098
 H1                  1.86255228    0.87394752    0.00000101
'''

pa.atom = '''
 O2                  -1.41904607    0.10858435    0.00000006
 H2                  -1.76001996   -0.36619300   -0.76029109
 H2                  -1.76001996   -0.36619300    0.76029121
'''

dimer.basis = {'H2': gto.basis.parse('''
BASIS "ao basis" SPHERICAL PRINT
#BASIS SET: (6s,3p) -> [4s,3p]
H    S
     13.0100000              0.0196850              0.0000000
      1.9620000              0.1379770              0.0000000
      0.4446000              0.4781480              0.0000000
      0.1220000              0.5012400              1.0000000
H    S
      0.0297400              1.0000000
H    S
      0.00725                1.000000
H    P
      0.7270000              1.0000000
H    P
      0.1410000              1.0000000
H    P
      0.02730                1.000000
END'''),
'O2': gto.basis.parse('''
BASIS "ao basis" SPHERICAL PRINT
#BASIS SET: (11s,6p,3d) -> [5s,4p,3d]
O    S
  11720.0000000              0.0007100             -0.0001600              0.0000000
   1759.0000000              0.0054700             -0.0012630              0.0000000
    400.8000000              0.0278370             -0.0062670              0.0000000
    113.7000000              0.1048000             -0.0257160              0.0000000
     37.0300000              0.2830620             -0.0709240              0.0000000
     13.2700000              0.4487190             -0.1654110              0.0000000
      5.0250000              0.2709520             -0.1169550              0.0000000
      1.0130000              0.0154580              0.5573680              0.0000000
      0.3023000             -0.0025850              0.5727590              1.0000000
O    S
      0.0789600              1.0000000
O    S
      0.0206                 1.000000
O    P
     17.7000000              0.0430180              0.0000000
      3.8540000              0.2289130              0.0000000
      1.0460000              0.5087280              0.0000000
      0.2753000              0.4605310              1.0000000
O    P
      0.0685600              1.0000000
O    P
      0.0171                 1.000000
O    D
      1.1850000              1.0000000
O    D
      0.3320000              1.0000000
O    D
      0.0930                 1.000000
END'''),
'H1': 'augccpvdz',
'O1': 'augccpvdz',}


pd.basis = 'augccpvdz'

pa.basis = {'H2': gto.basis.parse('''
BASIS "ao basis" SPHERICAL PRINT
#BASIS SET: (6s,3p) -> [4s,3p]
H    S
     13.0100000              0.0196850              0.0000000
      1.9620000              0.1379770              0.0000000
      0.4446000              0.4781480              0.0000000
      0.1220000              0.5012400              1.0000000
H    S
      0.0297400              1.0000000
H    S
      0.00725                1.000000
H    P
      0.7270000              1.0000000
H    P
      0.1410000              1.0000000
H    P
      0.02730                1.000000
END'''),
'O2': gto.basis.parse('''
BASIS "ao basis" SPHERICAL PRINT
#BASIS SET: (11s,6p,3d) -> [5s,4p,3d]
O    S
  11720.0000000              0.0007100             -0.0001600              0.0000000
   1759.0000000              0.0054700             -0.0012630              0.0000000
    400.8000000              0.0278370             -0.0062670              0.0000000
    113.7000000              0.1048000             -0.0257160              0.0000000
     37.0300000              0.2830620             -0.0709240              0.0000000
     13.2700000              0.4487190             -0.1654110              0.0000000
      5.0250000              0.2709520             -0.1169550              0.0000000
      1.0130000              0.0154580              0.5573680              0.0000000
      0.3023000             -0.0025850              0.5727590              1.0000000
O    S
      0.0789600              1.0000000
O    S
      0.0206                 1.000000
O    P
     17.7000000              0.0430180              0.0000000
      3.8540000              0.2289130              0.0000000
      1.0460000              0.5087280              0.0000000
      0.2753000              0.4605310              1.0000000
O    P
      0.0685600              1.0000000
O    P
      0.0171                 1.000000
O    D
      1.1850000              1.0000000
O    D
      0.3320000              1.0000000
O    D
      0.0930                 1.000000
END'''),}


dimer.build()
pd.build()
pa.build()

dimer = dft.UKS(dimer)
pd = dft.UKS(pd)
pa = dft.UKS(pa)

# Defining the tuned LC-PBE* functional. LibXC provides a tunable LC-PBE* w/ OP correlation, so we subtract the OP correlation. 

dimer.xc = 'HYB_GGA_XC_LC_PBEOP, -1 * GGA_C_OP_PBE + PBE'
dimer._numint.omega = 0.516
dimer._numint.alpha = 0.0
dimer._numint.beta = 1.0

pd.xc = 'HYB_GGA_XC_LC_PBEOP, -1 * GGA_C_OP_PBE + PBE'
pd._numint.omega = 0.516
pd._numint.alpha = 0.0
pd._numint.beta = 1.0

pa.xc = 'HYB_GGA_XC_LC_PBEOP, -1 * GGA_C_OP_PBE + PBE'
pa._numint.omega = 0.516
pa._numint.alpha = 0.0
pa._numint.beta = 1.0

dimer.kernel()
pa.kernel()
pd.kernel()
# Calculate noscf basis to print orbital occupations in
noscf_orbitals = basis_utils.noscfbasis(dimer, pd, pa)

basis_utils.print2molden(dimer, filename='SCForbitals')
basis_utils.print2molden(dimer, filename='NOSCForbitals', mo_coeff=noscf_orbitals)
rt_water = rt_scf.RT_SCF(dimer,1,2000)

# Declare which observables to be calculated/printed
rt_water.observables.update(mo_occ=True, charge=True, energy=True, hirsh_charge=True)

# Create object for complex absorbing potential and add to rt object
# Arguments are: expconst, emin, prefac, maxval

CAP = MOCAP(0.5, 0.0477, 1.0, 100.0)
rt_water.add_potential(CAP)

# Remove electron (in SCF basis)
# Input the two water fragments for their charge to be calculated
rt_utils.excite(rt_water, 4)
rt_utils.input_fragments(rt_water, pd, pa)

# Start calculation, send in noscf_orbitals to print
rt_water.kernel(mo_coeff_print=noscf_orbitals)

# This input yields nearly identical dynamics to https://doi.org/10.1021/acs.jpclett.4c01146
# After about 25 fs there is some small drift, however we attribute these differences to purely numerical differences in our integration and building of the Fock matrix through PySCF.
