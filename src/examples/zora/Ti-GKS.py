#!/usr/bin/env python
"""
Simple ZORA test case, do this first.
"""

from pyscf import gto, scf, dft
from tides import rt_scf, rt_vapp
from tides.zora.relativistic import ZORA
from sapporo import sapporo
from tides.rt_scf import RT_SCF

# Build mol
mol = gto.M(
	verbose = 0,
	atom='Ti 0 0 0',
	basis=sapporo,
    charge=4)

# Build and run GHF object
ti = dft.GKS(mol)
ti.xc = 'PBE0'

ti.kernel()

print('SCF ENERGY BEFORE ZORA (AU): ', ti.e_tot)
zora_obj = ZORA(ti)
Hcore = zora_obj.get_zora_correction()

ti.get_hcore = lambda *args: Hcore

ti.kernel()

print('SCF ENERGY AFTER ZORA (AU): ', ti.e_tot)

