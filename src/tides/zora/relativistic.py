import numpy as np
import scipy.special
import pyscf.dft as dft
from tides.zora.modbas2c import modbas2c

'''
Zeroth-Order Regular Approximation (ZORA)

!!!Only use with generalized references!!!

Typical usage where `mol` is pyscf molecule object:

>>> mol = pyscf.gto.M(...)
>>> ti = pyscf.scf.ghf.GHF(mol)
>>> from tides.zora.relativistic import ZORA
>>> zora_obj = ZORA(mol)
>>> Hcore = zora_obj.get_zora_correction()
>>> ti.get_hcore = lambda *args: Hcore

Overwrite the pyscf get_hcore function.
'''

__author__ = 'Nathan Gillispie'

class ZORA():
    def __init__(self, scf):
        self._scf = scf
        self.molecule = scf.mol

    def _read_basis(self,atoms):
        """Reads the model potential basis `modbas.2c`"""

        basis_file = modbas2c.split("\n")
        # remove empty lines
        basis_file = [line for line in basis_file if line != '']

        self.c_a = []

        atoms = [a.lower() for a, _ in self.molecule._atom]
        lower_atoms = list(map(lambda a: a.lower(), atoms))
        for atom in lower_atoms:
            position = [line for line, a in enumerate(basis_file) if a.split()[0] == atom][0]
            nbasis = int(basis_file[position][10:15])
            array = np.loadtxt(basis_file[position+2:position+2+nbasis]).transpose((1,0))
            self.c_a.append((np.array(array[1]),np.sqrt(np.array(array[0]))))

    def compute_Veff(self):
        """
        Computes the points, weights and ZORA integration kernel
        """
        _atoms = self.molecule._atom
        grid = dft.gen_grid.Grids(self.molecule)

        # ranges from 3 to 9
        # TODO: add option
        grid.level = 8

        atomic_grid = grid.gen_atomic_grids(self.molecule)
        points, self.weights = grid.get_partition(self.molecule, atomic_grid)

        #get model potential
        Z = self.molecule.atom_charges()
        self._read_basis(_atoms)

        Vtot = np.zeros((len(points)))
        for Ci, C in enumerate(_atoms):
            PA = points - self.molecule.atom_coords()[Ci]
            RPA = np.sqrt(np.sum(PA**2, axis=1))
            c, a = self.c_a[Ci]
            outer = np.outer(a,RPA)
            Vtot += np.einsum("i,i,ip->p",c,a,scipy.special.erf(outer)/outer,optimize=True)
            Vtot -= Z[Ci]/RPA
        self.kernel = np.asarray(Vtot)

        self.points  = np.asarray(points)
        self.weights = np.asarray(self.weights)

        print("   ZORA grid computed successfuly!",flush=True)

    def get_zora_correction(self):
        print("    Computing ZORA integrals.",flush=True)

        self.compute_Veff()
        nbf = self.molecule.nao
        self.eps_scal_ao = np.zeros((nbf,nbf))
        self.T = np.zeros((4,nbf,nbf))

        npoints = len(self.points)
        print("    Number of grid points: %i"%npoints)
        # Hard coded batch size for memory reasons, if you're running out of memory, lower this.
        batch_size = 1024*1024
        excess = npoints%batch_size
        nbatches = (npoints-excess)//batch_size
        print("    Number of batches: %i"%(nbatches+1))
        print("    Maximum Batch Size: %i"%batch_size)
        print("    Memory estimation for ZORA build: %8.4f mb"%(batch_size*nbf*6*8/1024./1024.),flush=True)
        for batch in range(nbatches+1):
            low = batch*batch_size
            if batch < nbatches:
                high = low+batch_size
            else:
                high = low+excess

            bpoints  = self.points[low:high]
            bweights = self.weights[low:high]
            bVzora   = self.kernel[low:high]
            ao_val = dft.numint.eval_ao(self.molecule, bpoints, deriv=1)
            kernel = 1./(2.*(137.036**2) - bVzora)
            self.T[0] += np.einsum("xip,xiq,i->pq",ao_val[1:],ao_val[1:],bweights*kernel,optimize=True) * (137.036**2)
            self.eps_scal_ao += np.einsum("xip,xiq,i->pq",ao_val[1:],ao_val[1:],bweights*kernel**2,optimize=True) * (137.036**2)
            kernel = bVzora/(4.*(137.036**2) - 2.*bVzora)
            # x component
            self.T[1] += np.einsum("ip,iq,i->pq",ao_val[2],ao_val[3],bweights*kernel,optimize=True)
            self.T[1] -= np.einsum("ip,iq,i->pq",ao_val[3],ao_val[2],bweights*kernel,optimize=True)

            self.T[2] += np.einsum("ip,iq,i->pq",ao_val[3],ao_val[1],bweights*kernel,optimize=True)
            self.T[2] -= np.einsum("ip,iq,i->pq",ao_val[1],ao_val[3],bweights*kernel,optimize=True)

            self.T[3] += np.einsum("ip,iq,i->pq",ao_val[1],ao_val[2],bweights*kernel,optimize=True)
            self.T[3] -= np.einsum("ip,iq,i->pq",ao_val[2],ao_val[1],bweights*kernel,optimize=True)

        print("    ZORA integrals computed!\n",flush=True)

        self.H_so = np.zeros((2*nbf,2*nbf),dtype=complex)
        Kx = 1j * self.T[1]
        Ky = 1j * self.T[2]
        Kz = 1j * self.T[3]

        self.H_so[:nbf,:nbf] =   Kz
        self.H_so[nbf:,nbf:] =  -Kz
        self.H_so[:nbf,nbf:] =  (Kx - 1j*Ky)
        self.H_so[nbf:,:nbf] =  (Kx + 1j*Ky)
        T = np.zeros(np.shape(self.H_so))
        T[:nbf,:nbf] = self.T[0]
        T[nbf:,nbf:] = self.T[0]

        Vnuc = np.zeros(self.H_so.shape)
        _Vnuc = self.molecule.intor("int1e_nuc")
        Vnuc[:nbf,:nbf] = _Vnuc
        Vnuc[nbf:,nbf:] = _Vnuc

        return self.H_so + T + Vnuc

