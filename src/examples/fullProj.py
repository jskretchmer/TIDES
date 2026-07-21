import sys
import os
from pyscf import gto, scf, ao2mo, mcscf

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

import tdfci as tdfci
import make_hams as make_hams
import fci_mod as fci_mod
import utils as utils
import rt_cas_ras
import numpy as np
from rt_scf import RT_SCF

# NOTE: generalized TDFCI with generalized static FCI calculation
'''
n_threads = os.environ.get('SLURM_CPUS_PER_TASK', '12')
os.environ['MKL_NUM_THREADS'] = n_threads
os.environ['OPENBLAS_NUM_THREADS'] = n_threads
os.environ['OMP_NUM_THREADS'] = '1'
os.environ['MKL_DYNAMIC'] = 'FALSE'
'''
NL = 3
NR = 4
Ndots = 1
Nsites = NL + NR + Ndots
Nele = np.copy(Nsites)

timp = 0.4
timplead = 0.4
tleads = 1.0
Vg = 0.0
Full = True

delt = 0.001
Nstep = 2000
#Nstep = 5
Nprint = 20
boundary = False
Ntot = 10

# Initital Restricted Static Calculation
U = 0.0
Vbias = 0.0
gen = False
casSize = 2

h_site, V_site = make_hams.make_ham_multi_imp_anderson_realspace(
    Ndots, NL, NR, Vg, U, timp, timplead, Vbias, tleads, boundary, Full
)

CIcoeffs,coeff, tdcas = fci_mod.FCI_GS(h_site, V_site, 0.0, Nsites, Nele, casSize,gen)
#coeff, CASCIcoeffs = fci_mod.FCI_GS(h_site, V_site, 0.0, Nsites, Nele, gen)
#print(CIcoeffs)
#print(type(CASCIcoeffs))
#CIcoeffs = fci_mod.FCI_GS(h_site, V_site, 0.0, Nsites, Nele, gen)


# Initializing Dynamics Calculation
U = 1.0
Vbias = 0.0
gen = False

h_site, V_site = make_hams.make_ham_multi_imp_anderson_realspace(
    Ndots, NL, NR, Vg, U, timp, timplead, Vbias, tleads, boundary, Full
)
'''
tdfci = tdfci.tdfci(
    Nsites, Nele, h_site, V_site, CIcoeffs, delt, Nstep, Nprint, 0.0, gen
)
tdfci.kernel()
#sys.exit()


tdhf.get_hcore = lambda *args: h_site
tdhf._eri = ao2mo.restore(8, V_site, Nsites)
rt_tdhf = RT_SCF(tdhf,delt,Nstep*delt,'tdhfVoCheck.dat','tdhfVcCheck.dat',prop='rk4')
rt_tdhf.kernel()
#sys.exit()
'''
mol = gto.M()
mol.nelectron = Nele
mol.nao = Nsites
mol.spin = 0
mf = scf.RHF(mol)
mf.get_hcore = lambda *args: h_site
mf.get_ovlp = lambda *args: np.eye(Nsites)
mf._eri = ao2mo.restore(8, V_site, Nsites)

# CASSCF gs
mf.mo_coeff = tdcas.mo_coeff
toRun = mcscf.CASSCF(mf,casSize,casSize)
toRun.ci = tdcas.ci
'''
# CASCI gs
mf.mo_coeff = coeff
toRun = mcscf.CASCI(mf,casSize,casSize)
toRun.ci = CASCIcoeffs
toRun.mo_coeff = coeff

cas = rt_cas_ras.RT_CAS_RAS(
    toRun,delt,Nstep*delt,"outputCASCIOcigs.dat","corr_densityCASCIVcigs.dat",ovlp=np.eye(Nsites),h2e=V_site
)
cas.kernel()
sys.exit()

mol = gto.M()
mol.nelectron = Nele
mol.nao = Nsites
# this call is necessary to use user defined hamiltonian in fci step
mol.incore_anyway = True
mf = scf.RHF(mol)
mf.get_hcore = lambda *args: h_site
mf.get_ovlp = lambda *args: np.eye(Nsites)
mf._eri = ao2mo.restore(8, V_site, Nsites)
mf.kernel()

tdcas.get_hcore = lambda *args: h_site
tdcas.get_ovlp = lambda *args: np.eye(Nsites)
tdcas._eri = ao2mo.restore(8, V_site, Nsites)
ci0 = np.zeros(np.shape(CIcoeffs),dtype=np.complex128)
ci0[0][0] = 1.0
tdcas.ci = np.copy(ci0)
#sys.exit()
'''
casscf = rt_cas_ras.RT_CAS_RAS(
    'CASSCF',toRun,delt,Ntot,"u8o2p10c.dat","u8c2p10c.dat",1e-5,ovlp=np.eye(Nsites),h2e=V_site
)
casscf.kernel()
'''

toRun = mcscf.CASCI(mf,casSize,casSize)
tdcas.get_hcore = lambda *args: h_site
tdcas.get_ovlp = lambda *args: np.eye(Nsites)
tdcas._eri = ao2mo.restore(8, V_site, Nsites)
ci0 = np.zeros(np.shape(CIcoeffs),dtype=np.complex128)
ci0[0][0] = 1.0
tdcas.ci = np.copy(ci0)
casscf = rt_cas_ras_no_proj.RT_CAS_RAS(
    'CASSCF',tdcas,delt,Nstep*delt,"tdcasscfUVo4.dat","tdcasscfUVc4.dat",ovlp=np.eye(Nsites),h2e=V_site
)
casscf.kernel()
'''
