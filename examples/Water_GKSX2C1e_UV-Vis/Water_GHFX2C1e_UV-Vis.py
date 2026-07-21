import numpy as np
from pyscf import gto, scf, dft
from tides import rt_scf

'''
An example of defining a custom field. And how to set up an x2c() calculation.

In this case we will define a electric delta impulse field that works with a generalized spin + x2c object.
We must calculate the transition dipole tensor of our system using the x2c picture change. 
Luckily, PySCF has to do this to calculate dipole moments.
So we will use the same lines of code used in PySCF's "dip_moment" function for the X2C1E_GSCF class.
'''

mol = gto.M(
    verbose = 0,
    atom='''
  O     0.00000000    -0.00001441    -0.34824012
  H    -0.00000000     0.76001092    -0.93285191
  H     0.00000000    -0.75999650    -0.93290797
  ''',
    basis='6-31G',
    spin = 0)

mf = scf.GHF(mol).x2c()
mf.kernel()

class x2cDeltaField:
    def __init__(self, rt_scf, amplitude, center=0):

        self.amplitude = np.array(amplitude)
        self.center = center
        mol = rt_scf._scf.mol
        charges = mol.atom_charges()
        coords = mol.atom_coords()
        nuc_charge_center = np.einsum('z,zx->x', charges, coords) / charges.sum()
        mol.set_common_orig_(nuc_charge_center)
        # For X2C property operators must be rotated by the "picture change" associated with 4c -> 2c
        # See https://github.com/pyscf/pyscf/blob/master/examples/x2c/10-picture_change.py
        tdip = -1 * rt_scf._scf.with_x2c.picture_change(('int1e_r_spinor',
                                                       'int1e_sprsp_spinor'))
        self.tdip = tdip

    def calculate_field_energy(self, rt_scf):
        if rt_scf.current_time == self.center:
            return self.amplitude
        else:
            return [0, 0, 0]

    def calculate_potential(self, rt_scf):
        energy = self.calculate_field_energy(rt_scf)
        return -1 * np.einsum('xij,x->ij', self.tdip, energy)

rt_mf = rt_scf.RT_SCF(mf, 0.5, 2000)
rt_mf.observables.update(energy=True, dipole=True)

delta_field = x2cDeltaField(rt_mf, [0.0001, 0.0001, 0.0001])

rt_mf.add_potential(delta_field)

rt_mf.kernel()

