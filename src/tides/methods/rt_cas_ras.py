import numpy as np
import os
import sys
from scipy.linalg import inv
from pyscf import mcscf
from pyscf.fci import cistring
from pyscf.scf import addons
from pyscf.lib import logger
from tides.propagators.rt_casprop import propagate
from tides.observables import rt_observables
from tides.utils import fci_mod
from tides.utils.rt_utils import restart_from_chkfile

'''
TD-RAS-SCF is not implemented
Projects out virtual space
'''

def _build_ras_level_blocks(row_sums, col_sums, opts):
    '''
    Precomputes, for the RAS excitation-level restriction, which beta strings
    are allowed for each alpha excitation level and vice versa, as CSR-style
    (flat, offset) arrays. Every (lA, lB) block of the na x nb grid
    is either fully allowed or fully disallowed -- this lets contract_2e_complex_ras
    skip disallowed blocks entirely.
    '''
    alpha_level = row_sums.astype(np.int64)
    beta_level = col_sums.astype(np.int64)
    allowed_totals = set(opts) | {0}

    max_levelA = int(alpha_level.max()) if len(alpha_level) else 0
    max_levelB = int(beta_level.max()) if len(beta_level) else 0

    alpha_order = np.argsort(alpha_level, kind='stable')
    alpha_level_sorted = alpha_level[alpha_order]
    alpha_offset_by_level = np.searchsorted(alpha_level_sorted, np.arange(max_levelA + 2))

    beta_order = np.argsort(beta_level, kind='stable')
    beta_level_sorted = beta_level[beta_order]
    beta_offset_by_level = np.searchsorted(beta_level_sorted, np.arange(max_levelB + 2))

    allowed_beta_parts = []
    allowed_beta_offset = np.zeros(max_levelA + 2, dtype=np.int64)
    for LA in range(max_levelA + 1):
        parts = [
            beta_order[beta_offset_by_level[LB]:beta_offset_by_level[LB + 1]]
            for LB in range(max_levelB + 1)
            if (LA + LB) in allowed_totals
        ]
        seg = np.concatenate(parts) if parts else np.array([], dtype=np.int64)
        allowed_beta_parts.append(seg)
        allowed_beta_offset[LA + 1] = allowed_beta_offset[LA] + len(seg)
    allowed_beta_flat = (
        np.concatenate(allowed_beta_parts) if allowed_beta_parts else np.array([], dtype=np.int64)
    ).astype(np.int64)

    allowed_alpha_parts = []
    allowed_alpha_offset = np.zeros(max_levelB + 2, dtype=np.int64)
    for LB in range(max_levelB + 1):
        parts = [
            alpha_order[alpha_offset_by_level[LA]:alpha_offset_by_level[LA + 1]]
            for LA in range(max_levelA + 1)
            if (LA + LB) in allowed_totals
        ]
        seg = np.concatenate(parts) if parts else np.array([], dtype=np.int64)
        allowed_alpha_parts.append(seg)
        allowed_alpha_offset[LB + 1] = allowed_alpha_offset[LB] + len(seg)
    allowed_alpha_flat = (
        np.concatenate(allowed_alpha_parts) if allowed_alpha_parts else np.array([], dtype=np.int64)
    ).astype(np.int64)

    return (alpha_level, beta_level, allowed_beta_flat, allowed_beta_offset,
            allowed_alpha_flat, allowed_alpha_offset)


class RT_CAS_RAS:
    '''
    opt: String
        opt='CASCI' for TD-CAS/RAS-CI calculation
        opt='CASSCF' for TD-CAS/RAS-SCF calculation
    ras: pyscf mcscf CASSCF/CASCI object, solved for its t=0 state
    timestep: as defined for the given propogation method
    max_time: total time to run the dynamics for
    reg: Float, minimum allowed eigenvalue of matrices before inversion can take place
        Only actually effects CAS/RAS SCF
    opts: Array of integers indicating allowed excitations (1 allows -S, 2 allows -D, etc.). Full CAS calculation is opts left as default.
    filename: Name of output file
    h1e: 1 e Hamiltonian in AO basis at initial time
    h2e: 2 e Hamiltonian in AO basis at initial time
    prop: Propogation method in rt_integrators to use.
        rk4cr: Fourth-order runge-kutta integrator
        vv: Second-order symplectic split operator integrator. Only implemented for CASCI, not CASSCF
    frequency: How many time steps you want between prints to output files
    mo_to_ao: MO to AO transformation matrix (MO coefficient matrix). Sort columns (core,active). Do not include columns for virtual orbitals.
    orth: Orthogonal AO coefficient matrix
    chkfile: I don't use this, left to keep consistent with rt_scf class
    verbose: I don't use this, left to keep consistent with rt_scf class
    ovlp: AO overlap matrix
    '''
    def __init__(self, opt,ras, timestep, max_time, reg=1e-8,opts = None,filename=None, h1e=None, h2e=None, prop=None, frequency=1, mo_to_ao=None, orth=None, chkfile=None, verbose=3, ovlp=None):
        self.timestep = timestep
        self.frequency = frequency
        self.max_time = max_time

        # Initial CASCI/CASSCF state
        # Will be used to store dynamic attributes
        self._scf = ras

        self.ovlp = ovlp

        # Regularization parameter
        self.ep = reg
        
        # Determines whether TD-_-CI or TD-_-SCF will be performed
        self._castype = opt

        # Core + Active Orbital Space Size
        self.numP = self._scf.ncore + self._scf.ncas

        # Number of spin up electrons
        self.neleca = self._scf.nelecas[0] + self._scf.ncore

        # Number of spin down electrons
        self.nelecb = self._scf.nelecas[1] + self._scf.ncore

        # String excitation tables for the active space, cached since (ncas, nelecas)
        # is fixed for the whole propagation and apply_ham_pyscf_check is called
        # many times per timestep
        self.link_index = (
            cistring.gen_linkstr_index(range(self._scf.ncas), self._scf.nelecas[0]),
            cistring.gen_linkstr_index(range(self._scf.ncas), self._scf.nelecas[1]),
        )

        self.verbose = verbose

        # The time-dependent portion of the Hamiltonian will be stored here as a function, if it exists
        self._potential = []

        self.labels = [self._scf.mol._atom[idx][0] for idx, _ in enumerate(self._scf.mol._atom)]

        # Get Hamiltonians in AO basis, MO/AO basis transformation matrix, AO overlap matrix, and/or AO/OAO basis transformation matrix from cas object if no custom ones are given
        if h1e is None:
            self._h1e_AO_0 = self._scf.get_hcore().T
        else:
            self._h1e_AO_0 = h1e.T
        if h2e is None:
            self._h2e_AO = self._scf.mol.intor('int2e')
        else:
            self._h2e_AO = h2e
        if mo_to_ao is None:
            self.mo_to_ao = self._scf.mo_coeff[:,:self.numP]
        else:
            self.mo_to_ao = mo_to_ao
        if ovlp is None:
            self.ovlp = self._scf.mol.intor('int1e_ovlp')
        else:
            self.ovlp = ovlp
        if self._castype == 'CASSCF':
            if orth is None:
                self.orth = addons.canonical_orth_(self.ovlp)
            else:
                self.orth = orth
            self.orth_inv = inv(self.orth)

        if prop is None: prop = 'rk4cr'
        # See vv in rt_integrators
        if prop == 'vv':
            self.pMinusHalf = 0
            self.pDotH = 0
            self.firstStep = True
            if self._castype == 'CASSCF':
                print('TD-_-SCF isnt applicable to velocity verlet')
                sys.exit()
                # Will replace with error when error class gets implemented
        self.prop = prop

        # Number of atomic orbitals
        self.no = len(self._h1e_AO_0)

        # Zero matrix in TD-CAS-CI and TD-RAS-CI
        if self._castype == 'CASCI':
            self._zero_xMat = (
                np.zeros((self.numP, self.numP), dtype=np.complex128),
                np.zeros((self.no, self.numP), dtype=np.complex128),
            )

        # AO to MO transformation matrix
        self.ao_to_mo = self.get_ao_to_mo()

        if filename is None:
            self._log = logger.Logger(verbose=self.verbose)
        else:
            self._fh = open(filename, 'a') # Temporarily making _fh append to file
            self._log = logger.Logger(self._fh, verbose=self.verbose)

        self.casrdm1, self.casrdm2 = self.get_casrdm12()
        self.den_ao = self.get_den_ao()

        if len(np.shape(self.den_ao)) == 3:
            self.nmat = 2
        else:
            self.nmat = 1

        # Restart from chkfile, or create a chkfile
        # If restarting from chkfile, self.den_ao will be rewritten
        # I do not use this functionality and have not tested it
        self.chkfile = chkfile
        if chkfile is not None:
            if os.path.exists(self.chkfile):
                restart_from_chkfile(self)
                self.den_ao = mcscf.make_rdm1(self._scf)
            else:
                self.current_time = 0
        else:
            self.current_time = 0

        self.ras = False
        # self.ras being True indicates RAS CI/SCF
        # self.ras being False indicates CAS CI/SCF
        if opts is not None:
            self.opts = opts
            self.ras = True
            indA = np.array(cistring.make_strings(range(self._scf.ncas),self._scf.nelecas[0]),dtype=np.uint8)
            rowOccs = np.unpackbits(indA[:,np.newaxis],axis=1,bitorder='little',count=self._scf.ncas)
            nA = np.sum(rowOccs[0])
            p2A = np.delete(rowOccs,np.s_[0:nA],axis=1)
            indB = np.array(cistring.make_strings(range(self._scf.ncas),self._scf.nelecas[1]),dtype=np.uint8)
            colOccs = np.unpackbits(indB[:,np.newaxis],axis=1,bitorder='little',count=self._scf.ncas)
            nB = np.sum(colOccs[0])
            p2B = np.delete(colOccs,np.s_[0:nB],axis=1)
            row_sums = np.sum(p2A, axis=1)
            col_sums = np.sum(p2B, axis=1)
            self.ras_blocks = _build_ras_level_blocks(row_sums, col_sums, self.opts)

        self._t0 = self.current_time

        rt_observables._init_observables(self)

    def get_full_e(self,mo_coeff=None,civec=None):
        if mo_coeff is None: mo_coeff=self._scf.mo_coeff
        if civec is None: civec=self._scf.ci
        h1eff, energy_core = self._scf.get_h1eff(mo_coeff)
        eri_cas = self._scf.get_h2eff(mo_coeff)
        e_cas = self._scf.fcisolver.energy(h1eff,eri_cas,civec,self._scf.ncas,self._scf.nelecas)
        return energy_core+e_cas, e_cas

    # I don't use this, left to keep consistent with rt_scf class
    def istype(self, type_code):
        if isinstance(type_code, type):
            return isinstance(self, type_code)

        return any(type_code == t.__name__ for t in self.__class__.__mro__)
    
    # Increment current time by an amount determined by the propogation method
    def update_time(self):
        if self.prop == 'rk4cr' or self.prop == 'vv':
            self.current_time += (self.timestep/2)
        else:
            self.current_time += self.timestep

    # Updates the 1e Hamiltonian to reflect the current time
    def updateHam(self,h1eAO):
        if self._castype == 'CASSCF':
            self._h1e_orth = self.get_h1e_orth(h1eAO)
        self._h1e_mo = self.get_h1e_mo(h1eAO)

    # Updates MO coefficients to reflect the current time.
    #   Will also run updateHam if Hamiltonian is time dependent
    def updateMO(self,moNew,h1eAO):
        self.mo_to_ao = np.copy(moNew)
        self.ao_to_mo = self.get_ao_to_mo()
        self._scf.mo_coeff[:,:self.numP] = np.copy(self.mo_to_ao)
        if len(self._potential) > 0:
            self.updateHam(h1eAO)
        else:
            self._h1e_mo = self.get_h1e_mo(h1eAO)
        self._h2e_mo = self.get_h2e_mo()
        self.mo_to_orth = self.get_mo_to_orth()
        self.orth_to_mo = self.mo_to_orth.conj().T

    # Get AO to MO transformation matrix from MO to AO transformation matrix
    def get_ao_to_mo(self):
        return self.mo_to_ao.conj().T
    
    # Get MO to AO transformation matrix from AO to MO transformation matrix
    def get_mo_to_ao(self):
        return self.ao_to_mo.conj().T

    # Given 1e Hamiltonian in AO basis, transform to MO basis
    def get_h1e_mo(self,h1):
        return np.matmul(self.ao_to_mo,np.matmul(h1,self.mo_to_ao)).astype(np.complex128)

    # Given 1e Hamiltonian in AO basis, transform to OAO basis
    def get_h1e_orth(self,h1):
        return np.matmul(self.orth_inv,np.matmul(h1,self.orth)).astype(np.complex128)

    # Contracts the AO 2e tensor, flattened to (no, no^3), against a matrix.
    #   fromLeft=False:  h2eFlat @ mat        (mat has no^3 rows)
    #   fromLeft=True:   mat.T   @ h2eFlat    (mat has no  rows)
    def _contract_h2e(self, mat, fromLeft=False):
        h2eFlat = self._h2e_AO.reshape(self.no, -1)
        if np.iscomplexobj(self._h2e_AO) or not np.iscomplexobj(mat):
            return (mat.T @ h2eFlat) if fromLeft else (h2eFlat @ mat)
        if fromLeft:
            return (mat.real.T @ h2eFlat) + 1j * (mat.imag.T @ h2eFlat)
        return (h2eFlat @ mat.real) + 1j * (h2eFlat @ mat.imag)

    # Transform 2e Hamiltonian from AO to MO basis
    #   h2e_mo[a,b,c,d] = Σ_pqrs mo_to_ao[p,a]* mo_to_ao[q,b] mo_to_ao[r,c]* mo_to_ao[s,d]
    #                            * h2e_AO[p,q,r,s]
    # May want to revisit whether C++ based would be faster
    def get_h2e_mo(self):
        moConj = self.mo_to_ao.conj()
        # First transform is the expensive one (no^4 * numP); see _contract_h2e
        t1 = self._contract_h2e(moConj, fromLeft=True)
        t1 = t1.reshape(-1, self.no, self.no, self.no)                       # a,q,r,s
        t2 = np.einsum('aqrs,qb->abrs', t1, self.mo_to_ao, optimize=True)    # a,b,r,s
        t3 = np.einsum('abrs,rc->abcs', t2, moConj, optimize=True)           # a,b,c,s
        return np.einsum('abcs,sd->abcd', t3, self.mo_to_ao,
                         optimize=True).astype(np.complex128)
    
    # Transform 2e Hamiltonian from AO to OAO basis
    # Dead code for the moment, not optimized
    def get_h2e_orth(self):
        mat1 = np.einsum('ap,pqrs,qb',self.orth_inv,self._h2e_AO,self.orth).astype(np.complex128)
        return np.einsum('cr,abrs,sd',self.orth_inv,mat1,self.orth).astype(np.complex128)
    
    # Transform a single-particle orbital from MO to AO basis
    def rotate_mo_to_ao(self,coeff_mo):
        return np.matmul(self.mo_coeff_canon,coeff_mo)
    
    # Transform a single-particle orbital from AO to OAO basis
    def rotate_ao_to_orth(self, coeff_ao):
        return np.matmul(self.orth_inv, coeff_ao)
    
    # Returns density matrix in AO basis
    def get_den_ao(self):
        corr1RDMmo = np.zeros((self.numP, self.numP), dtype=np.complex128)
        np.fill_diagonal(corr1RDMmo[:self._scf.ncore, :self._scf.ncore], 2)
        corr1RDMmo[self._scf.ncore:self.numP, self._scf.ncore:self.numP] = self.casrdm1
        self._mo_occ = np.diag(corr1RDMmo)
        return(self.mo_to_ao @ corr1RDMmo @ self.ao_to_mo)
    
    # fci_mod.get_corr12RDM wrapper
    def get_casrdm12(self):
        corr1RDM, corr2RDM = fci_mod.get_corr12RDM(self._scf.ci, self._scf.ncas, self._scf.nelecas)
        # 1RDMs are Hermitian by construction; get_xMat relies on this via eigh(casrdm1).
        # Catches a non-Hermitian corr1RDM here instead of it silently passing through eigh.
        assert np.allclose(corr1RDM, corr1RDM.conj().T, atol=1e-10), (
            "casrdm1 is not Hermitian"
        )
        return corr1RDM, corr2RDM

    # fci_mod.get_corr1RDM wrapper
    def get_casrdm1(self):
        corr1RDM = fci_mod.get_corr1RDM(self._scf.ci, self._scf.ncas, self._scf.nelecas)
        assert np.allclose(corr1RDM, corr1RDM.conj().T, atol=1e-10), (
            "casrdm1 is not Hermitian"
        )
        return corr1RDM
    
    # MO to OAO transformation matrix
    def get_mo_to_orth(self):
        return(self.orth_inv @ self.mo_to_ao)
    
    # Expresses molecular orbital in OAO basis
    def mo_unitvec_to_orth(self,mo):
        return self.mo_to_orth[:,mo]
    
    # Express molecular orbital in AO basis
    def mo_unitvec_to_ao(self,mo):
        return self.mo_to_ao[:,mo]
    
    # Adds some time-dependent term to the list of time-dependent Hamiltonian terms
    def add_potential(self, *args):
        for v_ext in args:
            self._potential.append(v_ext)

    # Updates h1e in AO basis to reflect Hamiltonian at the current time
    def apply_potential(self):
        toReturn = np.copy(self._h1e_AO_0)
        if len(self._potential) > 0:
            for v_ext in self._potential:
                toReturn = toReturn + v_ext.calculate_potential(self)
        return toReturn

    # Gets Q operator in OAO basis as defined in Phys. Rev. A 89, 063416
    def get_q_orth(self):
        p_orth = self.mo_to_orth[:, :self.numP]
        return np.eye(self.no) - np.einsum('ip,jp->ij', p_orth, p_orth.conj())
    
    # Gets Wab operator in OAO basis as defined in Phys. Rev. A 88, 023402
    # Note that this corresponds to p=a and q=b in eq 32
    def get_w_orth(self,a,b):
        aBra = self.mo_unitvec_to_ao(a).conj()
        bKet = self.mo_unitvec_to_ao(b)
        w_ao = np.matmul(aBra,np.matmul(self._h2e_AO,bKet))
        return np.matmul(self.orth_inv,np.matmul(w_ao,self.orth)).astype(np.complex128)

    # Returns QU block of virtual-space-unprojected X-matrix
    # See eq 36 of Phys. Rev. A 88, 023402
    def getQU(self,dBarInv):
        h2e_mo_term1 = self._h2e_mo[:self._scf.ncore,:self._scf.ncore,self._scf.ncore:,:self._scf.ncore]
        term1a = 4*np.einsum('vvku->ku',h2e_mo_term1)
        term1b = 2*np.einsum('vukv->ku',h2e_mo_term1)
        term1 = term1a-term1b
        h2e_mo_term2a = self._h2e_mo[self._scf.ncore:,self._scf.ncore:,self._scf.ncore:,:self._scf.ncore]
        h2e_mo_term2b = self._h2e_mo[self._scf.ncore:,:self._scf.ncore,self._scf.ncore:,self._scf.ncore:]
        term2a = 2*np.einsum('lmku,ml->ku',h2e_mo_term2a,self.casrdm1)
        term2b = np.einsum('lukm,ml->ku',h2e_mo_term2b,self.casrdm1)
        term2 = term2a-term2b
        term3 = np.einsum('jlmu,jlmk->ku',h2e_mo_term2a,self.casrdm2)
        h2e_mo_term4 = self._h2e_mo[self._scf.ncore:,:self._scf.ncore,:self._scf.ncore,:self._scf.ncore]
        term4a = 2*np.einsum('vvlu,kl->ku',h2e_mo_term1,self.casrdm1)
        term4b = np.einsum('lvvu,kl->ku',h2e_mo_term4,self.casrdm1)
        term4 = term4a-term4b
        termSum = term1+term2-term3-term4
        returnAdd = np.einsum('qk,ku->qu',dBarInv,termSum)
        hAdd = self._h1e_mo[self._scf.ncore:,:self._scf.ncore]
        return hAdd+returnAdd
    
    def get_xMat(self):
        '''
        toReturn: Returns R-matrix in Phys. Rev. A 88, 023402, ignoring rows/columns with virtual orbitals.
            For CI equations of motion
        toReturnAO: Returns columns of X|mo> in AO basis. Solves equations 38 and 39 in Phys. Rev. A 88, 023402
            For orbital equations of motion
        Refer to this paper for all equations in the comments
        '''
        if self._castype == 'CASCI':
            # R is the zero matrix in TD-CAS-CI and TD-RAS-CI
            return self._zero_xMat
        else:
            toReturn = np.zeros((self.numP,self.numP),dtype=np.complex128)
            toReturnAO = np.zeros((self.no,self.numP),dtype=np.complex128)
            ncore = self._scf.ncore
            numP = self.numP

            # Eq 37
            qMat = self.get_q_orth()

            # Regularization
            # Matrices to be inverted with eigenvalues below self.ep are set to self.ep
            # Both of these matrices are directly related to the 1RDM, so the 1RDM must be updated to reflect this
            eigvalrdm, eigvecrdm = np.linalg.eigh(self.casrdm1)
            eigvalrdm = np.clip(eigvalrdm.real, self.ep, 2 - self.ep)
            dInvEig = 1.0 / eigvalrdm
            dbarInvEig = 1.0 / (2 - eigvalrdm)
            eigMatRdm = np.diag(eigvalrdm)
            eigRdmInv = np.diag(dInvEig)
            eigDBarInv = np.diag(dbarInvEig)
            eigvecrdm_H = eigvecrdm.conj().T
            self.casrdm1 = eigvecrdm @ eigMatRdm @ eigvecrdm_H
            dInv = eigvecrdm @ eigRdmInv @ eigvecrdm_H
            dBarInv = eigvecrdm @ eigDBarInv @ eigvecrdm_H

            # Only R[ti] effects CI equations of motion
            quMat = self.getQU(dBarInv)
            toReturn[self._scf.ncore:self.numP,:self._scf.ncore] = np.copy(quMat)
            toReturn[:self._scf.ncore,self._scf.ncore:self.numP] = quMat.conj().T
            aoColForCore = np.matmul(self.mo_to_orth[:,self._scf.ncore:],quMat).T
            aoColForAct = np.matmul(self.mo_to_orth[:,:self._scf.ncore],quMat.conj().T).T

            mo_core_ao = self.mo_to_ao[:, :ncore]
            mo_act_ao = self.mo_to_ao[:, ncore:numP]

            # Core "density" and casrdm1-weighted active "density", both in AO basis
            P_core = np.einsum('qv,rv->qr', mo_core_ao.conj(), mo_core_ao)
            Q_act = np.einsum('ql,kl,rk->qr', mo_act_ao.conj(), self.casrdm1, mo_act_ao)

            stacked = np.stack([P_core, Q_act])  # (2, no, no)
            M1, M3 = np.einsum('pqrs,bqs->bpr', self._h2e_AO, stacked)
            M2, M4 = np.einsum('pqrs,bqr->bps', self._h2e_AO, stacked)

            w_core_diag = self.orth_inv @ M1 @ self.orth

            # M2 is shared: w_core_nondiag[m,u] = Σ_v (get_w_orth(v,u) @ mo_to_orth[:,v])[m]
            #               w_core_act_nondiag[m,q] = Σ_u (get_w_orth(u,qMO) @ mo_to_orth[:,u])[m]
            M2_orth = self.orth_inv @ M2
            w_core_nondiag = M2_orth @ mo_core_ao       # (no, ncore)
            w_core_act_nondiag = M2_orth @ mo_act_ao    # (no, ncas)

            # casrdm1-weighted analogues of w_core_diag/w_core_nondiag, for the core equations' Eq 29 term
            term3a = self.orth_inv @ M3 @ self.orth
            term3b = (self.orth_inv @ M4) @ mo_core_ao  # (no, ncore)

            # Eq 30's casrdm2/dInv-weighted term for the active equations.
            #   gTerm[a,b,c,t] = Σ_{j,m,k} mo_act_ao[a,j]* mo_act_ao[b,m] mo_act_ao[c,k]
            #                              * casrdm2_dInv[m,j,k,t]
            #   term2[p,t]     = Σ_{a,b,c} h2e_AO[p,a,b,c] gTerm[a,b,c,t]
            casrdm2_dInv = np.einsum('lmjk,lq->mjkq', self.casrdm2, dInv)
            gTerm = np.einsum('aj,mjkt->amkt', mo_act_ao.conj(), casrdm2_dInv, optimize=True)
            gTerm = np.einsum('bm,amkt->abkt', mo_act_ao, gTerm, optimize=True)
            gTerm = np.einsum('ck,abkt->abct', mo_act_ao, gTerm, optimize=True)
            gTerm = gTerm.reshape(-1, self._scf.ncas)
            term2 = self.orth_inv @ self._contract_h2e(gTerm)  # (no, ncas)

            mo_orth = self.mo_to_orth[:, :self.numP]

            # Eq 38 solves core equations of motion
            u_orth = mo_orth[:, :ncore]
            virt_all = self._h1e_orth.T @ u_orth  # (no, ncore)
            virt_all += 2 * w_core_diag @ u_orth  # (no, ncore)
            virt_all -= w_core_nondiag
            virt_all += term3a @ u_orth
            virt_all -= 0.5 * term3b
            aoCol_all = qMat @ virt_all
            aoCol_all += aoColForCore.T
            toReturnAO[:, :ncore] = self.orth @ aoCol_all

            # Eq 39 solves active orbital equations of motion
            mo_act = mo_orth[:, ncore:numP]
            h1e_act = self._h1e_orth.T @ mo_act  # (no, ncas)
            virt_act = h1e_act + term2  # (no, ncas)
            virt_act += 2 * w_core_diag @ mo_act  # (no, ncas)
            virt_act -= w_core_act_nondiag
            aoCol_act = qMat @ virt_act + aoColForAct.T
            toReturnAO[:, ncore:numP] = self.orth @ aoCol_act

            return toReturn, toReturnAO
    
    # Returns active space constant terms and 1e and 2e Hamiltonians
    # Analogous to final result on page 12 of group's dmet/dmet_original_jctc notes
    def get_actH(self,xMat):
        ncore = self._scf.ncore
        numP = self.numP
        h1 = self._h1e_mo - xMat
        Econst = 2 * np.sum(np.diag(h1))
        h2e_cc = self._h2e_mo[:ncore, :ncore, :ncore, :ncore]
        Econst += 2 * np.einsum('eeff->', h2e_cc) - np.einsum('effe->', h2e_cc)
        h1Mat = h1[ncore:numP, ncore:numP]
        h2e_ca = self._h2e_mo[ncore:numP, ncore:numP, :ncore, :ncore]
        h2e_cb = self._h2e_mo[ncore:numP, :ncore, :ncore, ncore:numP]
        h1Mat += 2 * np.einsum('baee->ab', h2e_ca) - np.einsum('beea->ab', h2e_cb)
        h2Mat = self._h2e_mo[ncore:numP, ncore:numP, ncore:numP, ncore:numP]
        return Econst, h1Mat, h2Mat

    # Begin time propogation. I don't use mo_coeff_print, left to keep consistent with rt_scf class
    def kernel(self, mo_coeff_print=None):
        try:
            propagate(self, mo_coeff_print)
        except Exception:
            raise
        finally:
            if np.isclose(self.current_time,self.max_time):  # So calculation terminates once max_time is reached after restarts
                self._log.note('Done')
            else:
                self._log.note('Propogation Stopped Early')
            if hasattr(self,'fh'):
                self.fh.close()
            if hasattr(self,'_xyz_fh'):
                # This is only important for unfrozen nuclei, printing .xyz files
                # Putting this here anyways for RT_Ehrenfest and other future derived classes
                self._xyz_fh.close()

        return self
