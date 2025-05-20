#!/usr/bin/env python
"""
Simple ZORA test case to show off ground-state references.
Real-time SCF can be trivially added at the end.
"""

import pyscf
from sapporo import sapporo
from tides.zora.relativistic import ZORA

mol = pyscf.gto.M(
	atom='Ti 0 0 0',
	basis=sapporo,
    charge=4,
	verbose=0,
)

def compute_e(ti):
    nr_e = ti.kernel()

    zora_obj = ZORA(ti)
    Hcore = zora_obj.get_zora_correction()
    ti.get_hcore = lambda *args: Hcore

    sr_e = ti.kernel()
    return nr_e, sr_e

print('GKS energy %.7f + ZORA --> %.7f'%compute_e(pyscf.dft.GKS(mol, xc='PBE0')))
print('GHF energy %.7f + ZORA --> %.7f'%compute_e(pyscf.scf.GHF(mol)))
print('UHF energy %.7f + ZORA --> %.7f'%compute_e(pyscf.scf.UHF(mol)))
print('RHF energy %.7f + ZORA --> %.7f'%compute_e(pyscf.scf.RHF(mol)))
