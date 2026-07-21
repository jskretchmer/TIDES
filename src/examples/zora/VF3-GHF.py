#!/usr/bin/env python
"""
ZORA test case with RT-SCF
"""

from pyscf import gto, scf, dft
from tides import rt_scf, rt_vapp
from tides.zora.relativistic import ZORA
from sapporo import sapporo
from tides.rt_scf import RT_SCF

# Build mol
mol = gto.M(
	verbose = 0,
	atom='V 0 0 0;F 1.86  0.00  0.00;F -0.93  1.61  0.00;F 0.93 -1.61 0.00',
    basis="ccpvdz",
    charge=2,
    spin=0,
)

ti = scf.ghf.GHF(mol)

ti.kernel()

print('SCF ENERGY BEFORE ZORA (AU): ', ti.e_tot)
zora_obj = ZORA(ti)
Hcore = zora_obj.get_zora_correction()

ti.get_hcore = lambda *args: Hcore

ti.kernel()

print('SCF ENERGY AFTER ZORA (AU): ', ti.e_tot)

rt_mf = RT_SCF(ti, 0.0001, 0.3)
rt_mf.observables.update(mag=True)

rt_mf.kernel()

