import numpy as np
from scipy.linalg import inv, eigh



'''
Molecular Orbital Complex Absorbing Potential
'''
# gotta change something so that the dimensions match up like the calculate potential function, in UKS the C_AO basis is 3D and is 2D for RKS

# we first define our parent class, in which we define our damping diagonal and we also form the OAO  CAP

class MOCAP:
	def __init__(self, expconst, emin, prefac=1, maxval=100):
                self.expconst = expconst
                self.emin = emin
                self.prefac = prefac
                self.maxval = maxval

        def calculate_cap(self, rt_scf, fock=None):
        #this here orthogonalizes the fock matrix (rt_scf,orth = X)
                if fock is None:
                        fock=rt_scf.fock_ao

                if rt_scf.nmat == 1:
                        return self._calculate_cap_single(rt_scf, fock)
                else:
                     	cap_0 = self._calculate_cap_single(rt_scf, fock[0])
                        cap_1 = self._calculate_cap_single(rt_scf, fock[1])
                        return np.stack([cap_0, cap_1])

        def _calculate_cap_single(self, rt_scf, fock):
                fock_orth = np.dot(rt_scf.orth.T, np.dot(fock, rt_scf.orth))
                mo_energy, _ = np.linalg.eigh(fock_orth)

        #now we work to construct the damping diagonal (D), with the CAP in OAO basis equal to -1j*C'@D@C'.T
                damping_diagonal = []
                for energy in mo_energy:
                        energy_corrected = energy - self.emin
                        if energy_corrected > 0:
                                damping_term = self.prefac * (1 - np.exp(self.expconst* energy_corrected))
                                if damping_term < (-1 * self.maxval):
                                        damping_term = -1 * self.maxval
                                damping_diagonal.append(damping_term)
                        else:
                             	damping_diagonal.append(0)

                damping_diagonal = np.array(damping_diagonal).astype(np.complex128)
                damping_matrix = np.diag(damping_diagonal)

                C_OAO = self.get_OAO_coeff(fock, rt_scf)        #I think this might be needed as the C' does not always come from fock

                damping_matrix_OAO = np.stack([
                        np.dot(C_OAO[0], np.dot(damping_matrix, C_OAO[0].T.conj())),
                        np.dot(C_OAO[1], np.dot(damping_matrix, C_OAO[1].T.conj()))
                ])
                transform = inv(rt_scf.orth.T)
                damping_matrix_AO = np.dot(transform, np.dot(damping_matrix, transform.T))
                return 1j * damping_matrix_AO

        def get_OAO_coeff(self, fock, rt_scf):
                raise NotImplementedError("Must choose a choice of basis for the CAP.")

        def calculate_potential(self, rt_scf):
                return self.calculate_cap(rt_scf)

#error with dimensionality currently comes from looking at the calculate_potential function


        #in the original cap, we give the cap in the AO basis instead of the OAO basis for some reason.

#Create the OAO CAP using the dimer basis.
class DIMER(MOCAP):
        def __init__(self, dimer, expconst, emin, prefac=1, maxval=100):
                super().__init__(expconst, emin, prefac, maxval)
                self.dimer = dimer

        def get_OAO_coeff(self, fock, rt_scf):
                C_AO = self.dimer.mo_coeff
                overlap_DIMER  = self.dimer.get_ovlp()
                eigvals, eigvecs = np.linalg.eigh(overlap_DIMER)
                #we probably should add something here to add a possible division by zero
                s_inv_sqrt = np.diag(1.0 / np.sqrt(eigvals))
                X = np.dot(eigvecs, np.dot(s_inv_sqrt, eigvecs.T.conj()))
        #Now we can use the lowdin orthogonalization  or canonical orthogonalization to get the OAO representation, in lowdin the transformation is C'=U@s^-0.5@U>
                if C_AO.ndim == 3: #if we have UKS
                        if fock is rt_scf.fock_ao[0]:
                                return np.dot(X, C_AO[0])
                        elif fock is rt_scf.fock_ao[1]:
                                return np.dot(X, C_AO[1])
                return np.dot(X, C_AO)

#Create the OAO CAP using the NOSCF basis.
class NOSCF(MOCAP):
        def __init__(self, dimer, noscf_orbitals, expconst, emin, prefac=1, maxval=100):
                super().__init__(expconst, emin, prefac, maxval)
                self.dimer = dimer
                self.noscf_orbitals = noscf_orbitals

        def get_OAO_coeff(self, fock, rt_scf):
                C_AO = self.noscf_orbitals
                overlap_dimer = self.dimer.get_ovlp()
                overlap_NOSCF = np.dot(C_AO.T.conj(), np.dot(overlap_dimer, C_AO))
                eigvals, eigvecs = np.linalg.eigh(overlap_NOSCF)
                s_inv_sqrt = np.diag(1.0 / np.sqrt(eigvals))
        #Now we can use the lowdin orthogonalization or canonical orthogonalization to get the OAO representation, in lowdin the transformation is C'=U@s^-0.5@U.>
                X = np.dot(eigvecs, np.dot(s_inv_sqrt, eigvecs.T.conj()))
                if C_AO.ndim == 3: #if we have UKS
                        if fock is rt_scf.fock_ao[0]:
                                return np.dot(X, C_AO[0])
                        elif fock is rt_scf.fock_ao[1]:
                                return np.dot(X, C_AO[1])
                return np.dot(X, C_AO)

#Create the OAO CAP using the basis that diagonalizes the time-dependent fock matrix.
class FORTHO(MOCAP):
        def __init__(self, expconst, emin, prefac=1, maxval=100):
                super().__init__(expconst, emin, prefac, maxval)
        def get_OAO_coeff(self, fock, rt_scf):
                fock_orth = np.dot(rt_scf.orth.T, np.dot(fock, rt_scf.orth))
                _, mo_coeff = np.linalg.eigh(fock_orth)
                return mo_coeff

