import numpy as np
from pyscf import gto, dft, scf, grad
from tides.grad_addons import complex_veff_
from tides import ehrenfest_force
from tides.rt_scf import RT_SCF
from tides.rt_nuclei import Nuc
from tides.rt_utils import _sym_orth, get_scf_orbitals
from tides.rt_utils import restart_from_chkfile
import os


'''
Real-time SCF + Ehrenfest
'''

class RT_Ehrenfest(RT_SCF):
    def __init__(self, scf, timestep, max_time, filename=None, prop=None, frequency=1, chkfile=None, verbose=3, Ne_step=10, N_step=10, get_mo_coeff_print=None):

        super().__init__(scf, timestep, max_time, filename, prop, frequency, None, chkfile, verbose)

        self.Ne_step = Ne_step
        self.N_step = N_step

        self.nuc = Nuc(self._scf.mol)
        # A bit redundant but recalling the restart_from_chkfile because nuc is not instantiated before super().__init__()
        if self.chkfile is not None:
            if os.path.exists(self.chkfile):
                restart_from_chkfile(self)

        self.den_ao = self._scf.make_rdm1(mo_occ=self.occ)
        if self.den_ao.dtype != np.complex128:
            self.den_ao = self.den_ao.astype(np.complex128)

        self.update_mol()

        if self._scf.istype('RKS'): self._grad_func = grad.RKS
        elif self._scf.istype('RHF'): self._grad_func = grad.RHF
        elif self._scf.istype('UKS'): self._grad_func = grad.UKS
        elif self._scf.istype('UHF'): self._grad_func = grad.UHF
        elif self._scf.istype('GKS'): self._grad_func = grad.UKS # grad.GKS doesn't exist
        elif self._scf.istype('GHF'): self._grad_func = grad.UHF # grad.GHF doesn't exist

        # Reminder to check if forces should be updated again after excite()
        self.nuc.force = ehrenfest_force.get_force(self)

        if get_mo_coeff_print is None:
            self.get_mo_coeff_print = get_scf_orbitals
        else:
            self.get_mo_coeff_print = get_mo_coeff_print

    def update_time(self):
        current_N = np.mod(round(self.current_time / self.timestep), self.Ne_step * self.N_step)
        current_Ne = np.mod(current_N, self.Ne_step)
        if current_N == 0: # k = 0, j = 0
            self.nuc.update_vel(0.5 * self.N_step * self.Ne_step * self.timestep)
            self.nuc.update_pos(0.5 * self.Ne_step * self.timestep)
            self.update_mol()
        elif current_Ne == 0: # k = 0, j != 0
            self.nuc.update_pos(0.5 * self.Ne_step * self.timestep)
            self.update_mol()
        if current_N == self.N_step * self.Ne_step - 1: # k = Ne_step-1, j = N_step-1
            self.nuc.update_pos(0.5 * self.Ne_step * self.timestep)
            self.update_mol()
            self.nuc.update_vel(0.5 * self.N_step * self.Ne_step * self.timestep)
        elif current_Ne == self.Ne_step - 1: # k = Ne_step-1, j != N_step-1
            self.nuc.update_pos(0.5 * self.Ne_step * self.timestep)
            self.update_mol()

        self.current_time += self.timestep

    def update_force(self):
        self.nuc.update_vel(-0.5 * self.N_step * self.Ne_step * self.timestep)
        self.nuc.force = ehrenfest_force.get_force(self)
        self.nuc.update_vel(0.5 * self.N_step * self.Ne_step * self.timestep)

    # Ehrenfest currently only supports symmetrical orthogonalization
    def update_mol(self):
        self._scf.reset(self.nuc.get_mol())
        self._scf.verbose = 0
        self.ovlp = self._scf.get_ovlp()
        # Generalized overlap is just stacks the restricted overlap, becomes an issue for gradients
        #self.ovlp = self._scf.mol.intor_symmetric('int1e_ovlp')
        self.evals, self.evecs = np.linalg.eigh(self.ovlp)
        self.orth = _sym_orth(self)

    def _update_grad(self):
        self._grad = self._scf.apply(self._grad_func)
        self._grad = complex_veff_(self._grad)

    def _get_omega(self):
        Rdot = self.nuc.vel
        mol = self._scf.mol
        dS = -mol.intor('int1e_ipovlp', comp=3)
        omega = np.zeros(self.ovlp.shape, dtype=np.complex128)
        aoslices = mol.aoslice_by_atom()
        for idx in range(mol.natm):
            p0, p1 = aoslices[idx,2:]
            #omega += np.einsum('x,xij,ik->jk', Rdot[idx], dS[:,p0:p1,:], X[p0:p1,:])
        #omega += np.matmul(RdSX, Xinv)
        return omega


    # Additional term arising from the moving nuclei in the classical path approximation to be added to the fock matrix
    # Reminder to turn the complex conserving potential term Omega into a potential classes
    # def _get_omega(self):
    #     mol = self._scf.mol
    #     Rdot = self.nuc.vel
    #     X = self.orth
    #     Xinv = inv(X)
    #     dS = -mol.intor('int1e_ipovlp', comp=3)
    # 
    #     Omega = np.zeros(X.shape, dtype = complex)
    #     RdSX = np.zeros(X.shape)
    #     aoslices = mol.aoslice_by_atom()
    #     for idx in range(mol.natm):
    #         p0, p1 = aoslices[idx,2:]
    #         RdSX += np.einsum('x,xij,ik->jk', Rdot[idx], dS[:,p0:p1,:], X[p0:p1,:])
    #     Omega += np.matmul(RdSX, Xinv)
    #     return Omega


