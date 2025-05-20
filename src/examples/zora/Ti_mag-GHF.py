#!/usr/bin/env python
"""
ZORA test case with RT-SCF.
Generalized is required for the magnetic dipole observable, but
ZORA also works with unrestricted and restricted references.
"""

import pyscf
from tides import rt_scf
from sapporo import sapporo

mol = pyscf.gto.M(
    atom='Ti 0 0 0',
    basis=sapporo,
    charge=4,
    verbose=0,
)

ti = pyscf.scf.GHF(mol)
ti.kernel()
print('SCF ENERGY BEFORE ZORA (AU): ', ti.e_tot)


### Steps to add ZORA to a TIDES calculation
## 1. Import ZORA
from tides.zora.relativistic import ZORA

## 2. Get the ZORA core Hamiltonian
zora_obj = ZORA(ti)
Hcore = zora_obj.get_zora_correction()

## 3. replace the pyscf core hamiltonian
ti.get_hcore = lambda *args: Hcore


ti.kernel()
print('SCF ENERGY AFTER ZORA (AU): ', ti.e_tot)

rt_mf = rt_scf.RT_SCF(ti, 0.0001, 0.3)
rt_mf.observables.update(mag=True)
rt_mf.kernel()

