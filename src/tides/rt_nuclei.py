import numpy as np
from pyscf import gto, dft, scf, data
from tides.basis_utils import _read_mol, _write_mol

'''
Nuclear Object for Real-Time Ehrenfest
'''

class Nuc:
    def __init__(self, mol):
        self._nnuc = len(mol._atom)
        self.basis, self.labels, pos = _read_mol(mol)
        self.pos = np.array(pos)
        #self.mass = 1836 * np.array([np.array(mol.atom_mass_list())[i] * np.ones(3) for i in range(self._nnuc)])
        self.mass = np.array([np.array([data.elements.COMMON_ISOTOPE_MASSES[m] * data.nist.AMU2AU for m in mol.atom_charges()])[i] * np.ones(3) for i in range(self._nnuc)])
        self.vel = np.zeros((self._nnuc, 3))
        self.force = np.zeros((self._nnuc, 3))
        self.spin = mol.spin
        self.charge = mol.charge

    def get_mol(self):
        return _write_mol(self.basis, self.labels, self.pos, self.spin, self.charge)

    def get_ke(self):
        return np.sum(0.5 * self.mass * self.vel**2, axis=1)

    def sample_vel(self, beta):
        self.vel = np.random.normal(scale = 1. / np.sqrt(beta * self.mass))

    # Position full step
    def update_pos(self, timestep):
        self.pos +=  self.vel * timestep

    # Velocity full step
    def update_vel(self, timestep):
        self.vel += self.force / self.mass * timestep

