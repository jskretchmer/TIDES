import numpy as np
from scipy.special import erf
import pyscf.dft as dft
from pyscf.data.nist import LIGHT_SPEED
from tides.zora.modbas2c import modbas2c

'''
Zeroth-Order Regular Approximation (ZORA)

Typical usage:

>>> mol = pyscf.gto.M(...)
>>> ti = pyscf.scf.GHF(mol)
>>> from tides.zora.relativistic import ZORA
>>> zora_obj = ZORA(mol)
>>> Hcore = zora_obj.get_zora_correction()
>>> ti.get_hcore = lambda *args: Hcore

Overwrite the pyscf get_hcore function and proceed as usual.
'''

__author__ = 'Nathan Gillispie'

class ZORA():
    def __init__(self, scf):
        self.molecule = scf.mol
        scf_types = [t.__name__ for t in scf.__class__.__mro__]
        if 'GHF' in scf_types:
            self._scf_type = 'GHF'
        elif 'UHF' in scf_types or 'RHF' in scf_types:
            self._scf_type = 'RHF'
        else:
            raise TypeError("ZORA: provided SCF object should require G/U/RHF core Hamiltonian")

    def get_model_potential(self, atoms):
        '''
        Reads the model potential basis `modbas.2c` and returns the contraction coefficients
        and square rooted exponents of the given atoms.
        '''

        basis_file = modbas2c.split('\n')
        basis_file = [line for line in basis_file if line != ''] # remove empty lines

        c_a = []

        atoms = [a.lower() for a, _ in self.molecule._atom]
        lower_atoms = list(map(lambda a: a.lower(), atoms))
        for atom in lower_atoms:
            position = [line for line, a in enumerate(basis_file) if a.split()[0] == atom][0]
            nbasis = int(basis_file[position][10:15])
            array = np.loadtxt(basis_file[position+2:position+2+nbasis]).transpose((1,0))
            # precompute the square root of the exponent
            c_a.append((np.array(array[1]),np.sqrt(np.array(array[0]))))

        return c_a

    def compute_zora_kernel(self):
        '''Returns the points, weights and ZORA integration kernel.'''
        _atoms = self.molecule._atom
        grid = dft.gen_grid.Grids(self.molecule)
        grid.level = 8    # 0-9
        grid.prune = None # Treutler + no pruning works the best

        atomic_grid = grid.gen_atomic_grids(self.molecule)
        points, weights = grid.get_partition(self.molecule, atomic_grid)

        Z = self.molecule.atom_charges()
        c_a = self.get_model_potential(_atoms)

        veff = np.zeros(points.shape[0])
        for Ci, C in enumerate(_atoms):
            PA = points - self.molecule.atom_coords()[Ci]
            RPA = np.sqrt(np.sum(PA**2, axis=1))
            c, a = c_a[Ci]
            outer = np.outer(a,RPA)
            veff += np.einsum('i,i,ip->p', c, a, erf(outer)/outer, optimize=True)
            veff -= Z[Ci]/RPA

        kernel = LIGHT_SPEED**2 / (2 * LIGHT_SPEED**2 - veff)

        return points, weights, kernel

    def get_zora_correction(self):
        points, weights, kernel = self.compute_zora_kernel()

        nbf = self.molecule.nao
        T = np.zeros((nbf,nbf))

        num_points = points.shape[0]
        batch_max_points = min(num_points, 1024**2) # Adjust according to memory availability / speed
        excess = num_points%batch_max_points
        num_full_batches = num_points // batch_max_points

        def batch_job(batch_slice):
            bpoints  = points[batch_slice]
            bweights = weights[batch_slice]
            bkernel = kernel[batch_slice]
            ao_val = dft.numint.eval_ao(self.molecule, bpoints, deriv=1)
            nonlocal T
            T += np.einsum("xip,xiq,i->pq", ao_val[1:], ao_val[1:], bweights*bkernel, optimize=True)

        for batch in range(num_full_batches):
            batch_job(slice(batch*batch_max_points, (batch+1)*batch_max_points))
        if excess > 0:
            batch_job(slice(batch_max_points * num_full_batches, None))

        _Vnuc = self.molecule.intor("int1e_nuc")
        if self._scf_type == 'GHF':
            # Scalar-relativistic core Hamiltonian
            h_core_sr = np.zeros((2*nbf,2*nbf), dtype=complex)
            h_core_sr[:nbf, :nbf] += T + _Vnuc
            h_core_sr[nbf:, nbf:] += T + _Vnuc
        elif self._scf_type == 'RHF':
            h_core_sr = T + _Vnuc

        return h_core_sr


