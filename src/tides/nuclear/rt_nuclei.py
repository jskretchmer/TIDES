import numpy as np
from pyscf import gto, dft, scf, data
from tides.utils.basis_utils import _read_mol, _write_mol

'''
Nuclear Object for Real-Time Ehrenfest
'''

class Nuc:
    def __init__(self, mol):
        self._nnuc = len(mol._atom)
        self.basis, self.labels, pos = _read_mol(mol)
        self.pos = np.array(pos)
        #self.mass = 1836 * np.array([np.array(mol.atom_mass_list())[i] * np.ones(3) for i in range(self._nnuc)])
        # Index the mass table by the element (true Z), not by mol.atom_charges(): with an ECP that is the
        # core-reduced charge and would give the mass of a different element. Ghost atoms get Z = 0, mass 0.
        Z = [gto.charge(mol.atom_symbol(i)) for i in range(self._nnuc)]
        self.mass = np.array([data.elements.COMMON_ISOTOPE_MASSES[z] * data.nist.AMU2AU * np.ones(3) for z in Z])
        self.vel = np.zeros((self._nnuc, 3))
        self.force = np.zeros((self._nnuc, 3))
        self.spin = mol.spin
        self.charge = mol.charge
        self._mol = mol.copy()

    def get_mol(self):
        # Move a copy of the original molecule so that everything else it carries
        # (ECPs, per-atom basis sets, ghost atoms, cart, nucmod, ...) is kept.
        mol = self._mol.copy()
        mol.set_geom_(self.pos, unit='Bohr')
        return mol

    def get_ke(self):
        return np.sum(0.5 * self.mass * self.vel**2, axis=1)

    def _inv_mass(self):
        # Massless centres (ghost atoms) do not move.
        return np.divide(1., self.mass, out=np.zeros_like(self.mass, dtype=float), where=self.mass > 0)

    def sample_vel(self, beta):
        self.vel = np.random.normal(scale = np.sqrt(self._inv_mass() / beta))

    # Position full step
    def update_pos(self, timestep):
        self.pos +=  self.vel * timestep

    # Velocity full step
    def update_vel(self, timestep):
        self.vel += self.force * self._inv_mass() * timestep

