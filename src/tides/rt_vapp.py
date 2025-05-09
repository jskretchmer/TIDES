import numpy as np
from pyscf import lib

'''
Real-time Time-Dependent Applied Potential (Sample Electric Field)
Currently for restricted/unrestricted
'''

class ElectricField:
    def __init__(self, field_type, amplitude, center=0,
                frequency=0, width=0, phase=0):

        # Some attributes are irrelevant depending on field type
        self.field_type = field_type
        self.amplitude = np.array(amplitude)
        self.center = center
        self.frequency = frequency
        self.width = width
        self.phase = phase

    def delta_energy(self):
        return self.amplitude

    def gaussian_energy(self, rt_scf):
        return (self.amplitude * ((np.exp(-1 * (rt_scf.current_time - self.center) ** 2 / 
        (2 * self.width ** 2))) * np.sin(self.frequency * rt_scf.current_time
                                        + self.phase)))

    def hann_energy(self, rt_scf):
        return (self.amplitude * ((np.sin(np.pi / self.width * 
        (rt_scf.current_time - self.center - self.width / 2))) ** 2 * 
        np.sin(self.frequency * rt_scf.current_time + self.phase)))

    def resonant_energy(self, rt_scf):
        return self.amplitude * np.sin(self.frequency * rt_scf.current_time + self.phase)

    def calculate_field_energy(self, rt_scf):
        if self.field_type == 'delta' and rt_scf.current_time == self.center:
            return self.delta_energy()

        elif self.field_type == 'gaussian':
            return self.gaussian_energy(rt_scf)

        elif self.field_type == 'hann':
            return self.hann_energy(rt_scf)

        elif self.field_type == 'resonant':
            return self.resonant_energy(rt_scf)

        else:
            return np.array([0.0, 0.0, 0.0])

    def calculate_potential(self, rt_scf):
        energy = self.calculate_field_energy(rt_scf)
        mol = rt_scf._scf.mol
        charges = mol.atom_charges()
        coords = mol.atom_coords()
        nuc_charge_center = np.einsum('z,zx->x', charges, coords) / charges.sum()
        mol.set_common_orig_(nuc_charge_center)

        if rt_scf._scf.istype('DKS') | rt_scf._scf.istype('DHF'):
            nmo = mol.nao_2c()
            tdip = np.zeros((3, nmo*2, nmo*2), dtype='complex128')
            c = lib.param.LIGHT_SPEED
            tdip[:,:nmo,:nmo] += -1 * mol.intor_symmetric('int1e_r_spinor', comp=3)
            tdip[:,nmo:,nmo:] += -1 * mol.intor_symmetric('int1e_sprsp_spinor', comp=3) / (2*c)**2
        elif rt_scf._scf.istype('X2C1E_GSCF'):
            tdip = -1 * rt_scf._scf.with_x2c.picture_change(('int1e_r_spinor', 'int1e_sprsp_spinor'))
        elif rt_scf._scf.istype('GHF'):
            tdip = np.tile(mol.intor_symmetric('int1e_r', comp=3), (1,2,2))
        else:
            tdip = -1 * mol.intor('int1e_r', comp=3)
        return -1 * np.einsum('xij,x->ij', tdip, energy)
