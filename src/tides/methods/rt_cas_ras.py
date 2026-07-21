import numpy as np
import os
import sys
from scipy.linalg import inv
from pyscf import mcscf, fci
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

class RT_CAS_RAS:
    '''
    opt: String
        opt='CASCI' for TD-CAS/RAS-CI calculation
        opt='CASSCF' for TD-CAS/RAS-SCF calculation
    ras: pyscf mcscf CASSCF/CASCI object, solved for its t=0 state
    timestep: as defined for the given propogation method
    max_time: total time to run the dynamics for
    outputName: name of output file used to check for numerical stability
    corrDenName: name of output file that shows each AO occupation at each time step
    reg: Float, minimum allowed eigenvalue of matrices before inversion can take place
        Only actually effects CAS/RAS SCF
    opts: Array of integers indicating allowed excitations (1 allows -S, 2 allows -D, etc.). Full CAS calculation is opts left as default.
    filename: I don't use this, left to keep consistent with rt_scf class
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
    def __init__(self, opt,ras, timestep, max_time, outputName, corrDenName, reg=1e-8,opts = None,filename=None, h1e=None, h2e=None, prop=None, frequency=1, mo_to_ao=None, orth=None, chkfile=None, verbose=3, ovlp=None):
        self.timestep = timestep
        self.frequency = frequency
        self.max_time = max_time

        # Initial CASCI/CASSCF state
        # Will be used to store dynamic attributes
        self._scf = ras

        self.ovlp = ovlp
        self.outName = outputName
        self.corName = corrDenName

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
        if orth is None:
            self.orth = addons.canonical_orth_(self.ovlp)
        else:
            self.orth = orth
        self.orth_inv = inv(self.orth)

        if prop is None: prop = 'rk4cr'
        # See vv in rt_integrators or rt_integrators_np
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
        self.no = len(self._h1e_AO)

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
        # self.ind gives CI vector indices that the input ras scheme allows to be nonzero
        # self.ras being True indicates RAS CI/SCF
        # self.ras being False indicates CAS CI/SCF
        if opts is not None:
            self.opts = opts
            self.ras = True
            indA = np.array(fci.cistring.make_strings(range(self._scf.ncas),self._scf.nelecas[0]),dtype=np.uint8)
            rowOccs = np.unpackbits(indA[:,np.newaxis],axis=1,bitorder='little',count=self._scf.ncas)
            nA = np.sum(rowOccs[0])
            p2A = np.delete(rowOccs,np.s_[0:nA],axis=1)
            indB = np.array(fci.cistring.make_strings(range(self._scf.ncas),self._scf.nelecas[1]),dtype=np.uint8)
            colOccs = np.unpackbits(indB[:,np.newaxis],axis=1,bitorder='little',count=self._scf.ncas)
            nB = np.sum(colOccs[0])
            p2B = np.delete(colOccs,np.s_[0:nB],axis=1)
            goodInd = np.zeros((len(p2A),len(p2B)))
            for i in range(len(goodInd)):
                for j in range(len(goodInd[0])):
                    el = np.sum(p2A[i]) + np.sum(p2B[j])
                    if el == 0 or el in self.opts:
                        goodInd[i,j] = 1
            self.ind = goodInd

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

    # Transform 2e Hamiltonian from AO to MO basis
    def get_h2e_mo(self):
        mat1 = np.einsum('ap,pqrs,qb',self.ao_to_mo,self._h2e_AO,self.mo_to_ao).astype(np.complex128)
        return np.einsum('cr,abrs,sd',self.ao_to_mo,mat1,self.mo_to_ao).astype(np.complex128)
    
    # Transform 2e Hamiltonian from AO to OAO basis
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
        corr1RDMmo = np.zeros((self.numP,self.numP)).astype(np.complex128)
        for a in range(self._scf.ncore):
            corr1RDMmo[a][a] = 2
        for a in range(self._scf.ncas):
            for b in range(self._scf.ncas):
                corr1RDMmo[a+self._scf.ncore][b+self._scf.ncore] = self.casrdm1[a][b]
        return(np.matmul(self.mo_to_ao,np.matmul(corr1RDMmo,self.ao_to_mo)))
    
    # fci_mod.get_corr12RDM wrapper
    def get_casrdm12(self):
        return fci_mod.get_corr12RDM(self._scf.ci, self._scf.ncas, self._scf.nelecas)
    
    # fci_mod.get_corr1RDM wrapper
    def get_casrdm1(self):
        return fci_mod.get_corr1RDM(self._scf.ci, self._scf.ncas, self._scf.nelecas)
    
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
        toReturn = np.eye(self.no)
        for p in range(self.numP):
            p_orth = self.mo_unitvec_to_orth(p)
            toReturn = toReturn - np.outer(p_orth,p_orth.conj())
        return toReturn
    
    # Gets Wab operator in OAO basis as defined in Phys. Rev. A 88, 023402
    # Note that this corresponds to p=a and q=b in eq 32
    def get_w_orth(self,a,b):
        aBra = self.mo_unitvec_to_ao(a).conj()
        bKet = self.mo_unitvec_to_ao(b)
        w_ao = np.matmul(aBra,np.matmul(self._h2e_AO,bKet))
        return np.matmul(self.orth_inv,np.matmul(w_ao,self.orth)).astype(np.complex128)
    
    # Returns X[q,u] for the X-matrix expressed in the MO basis
    # Corresponds to eq 36 of Phys. Rev. A 88, 023402
    # In group notes, corresponds to Xqu on page 9 of td_cas/td_casscf_notes
    def getQUOld(self,qn,un,dBarInv):
        toReturn = self._h1e_mo[qn+self._scf.ncore,un]
        for k in range(self._scf.ncas):
            kMO = k+self._scf.ncore
            prefac = dBarInv[qn,k]
            toAdd = 0
            for v in range(self._scf.ncore):
                toAdd = toAdd + 2*((2*self._h2e_mo[v][v][kMO][un])-self._h2e_mo[v][un][kMO][v])
                for l in range(self._scf.ncas):
                    lMO = l+self._scf.ncore
                    toAdd = toAdd - (self.casrdm1[k][l]*((2*self._h2e_mo[v][v][lMO][un])-self._h2e_mo[lMO][v][v][un]))
            for l in range(self._scf.ncas):
                for m in range(self._scf.ncas):
                    lMO = l+self._scf.ncore
                    mMO = m+self._scf.ncore
                    toAdd = toAdd + (self.casrdm1[m][l]*((2*self._h2e_mo[lMO][mMO][kMO][un])-self._h2e_mo[lMO][un][kMO][mMO]))
                    for j in range(self._scf.ncas):
                        jMO = j+self._scf.ncore
                        toAdd = toAdd - (self._h2e_mo[jMO][lMO][mMO][un]*self.casrdm2[j][l][m][k])
            toReturn = toReturn + (prefac*toAdd)
        return toReturn
    
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
        toReturn = np.zeros((self.numP,self.numP),dtype=np.complex128)
        toReturnAO = np.zeros((self.no,self.numP),dtype=np.complex128)
        if self._castype == 'CASCI':
            # R is the zero matrix in TD-CAS-CI and TD-RAS-CI
            return toReturn, toReturnAO
        else:
            # Eq 37
            qMat = self.get_q_orth()

            # Regularization
            # Matrices to be inverted with eigenvalues below self.ep are set to self.ep
            # Both of these matrices are directly related to the 1RDM, so the 1RDM must be updated to reflect this 
            eigvalrdm, eigvecrdm = np.linalg.eig(self.casrdm1)
            dInvEig = np.zeros(len(eigvalrdm))
            dbarInvEig = np.zeros(len(eigvalrdm))
            for eigInd in range(len(eigvalrdm)):
                if eigvalrdm[eigInd] < self.ep:
                    eigvalrdm[eigInd] = self.ep
                elif (2-eigvalrdm[eigInd]) < self.ep:
                    eigvalrdm[eigInd] = 2 - self.ep
                dInvEig[eigInd] = 1/eigvalrdm[eigInd]
                dbarInvEig[eigInd] = 1/(2-eigvalrdm[eigInd])
            eigMatRdm = np.diag(eigvalrdm)
            eigRdmInv = np.diag(dInvEig)
            eigDBarInv = np.diag(dbarInvEig)
            self.casrdm1 = eigvecrdm @ eigMatRdm @ inv(eigvecrdm)
            dInv = eigvecrdm @ eigRdmInv @ inv(eigvecrdm)
            dBarInv = eigvecrdm @ eigDBarInv @ inv(eigvecrdm)

            # Only R[ti] effects CI equations of motion
            quMat = self.getQU(dBarInv)
            toReturn[self._scf.ncore:self.numP][:self._scf.ncore] = np.copy(quMat)
            toReturn[:self._scf.ncore][self._scf.ncore:self.numP] = quMat.conj().T
            aoColForCore = np.matmul(self.mo_to_orth[:,self._scf.ncore:],quMat).T
            aoColForAct = np.matmul(self.mo_to_orth[:,:self._scf.ncore],quMat.T).T

            # Eq 38 solves core equations of motion
            for u in range(self._scf.ncore):
                uOrth = self.mo_unitvec_to_orth(u)

                # First part of eq 28
                virt = np.matmul(self._h1e_orth.T,uOrth)

                # Second part of eq 28. Note all core orbitals are dynamic core.
                for v in range(self._scf.ncore):
                    virt = virt + (2*np.matmul(self.get_w_orth(v,v),uOrth)) - (np.matmul(self.get_w_orth(v,u),self.mo_unitvec_to_orth(v)))
                
                # Eq 29
                for l in range(self._scf.ncas):
                    for k in range(self._scf.ncas):
                        lMO = l+self._scf.ncore
                        kMO = k+self._scf.ncore
                        virt = virt + (self.casrdm1[k,l]*(np.matmul(self.get_w_orth(lMO,kMO),uOrth)-(np.matmul(self.get_w_orth(lMO,u),self.mo_unitvec_to_orth(kMO))/2)))
                
                # Project into virtual space
                aoCol = np.matmul(qMat,virt)
                aoCol = aoCol + aoColForCore[u]

                '''
                # Now add active space contributions
                for q in range(self._scf.ncas):
                    qMO = q+self._scf.ncore
                    qu = quMat[q,u]
                    # qu represents R[ti]
                    aoCol = aoCol + (qu*self.mo_unitvec_to_orth(qMO))
                '''

                # Rotate to AO basis
                colFin = np.matmul(self.orth,aoCol)

                # Assign to output
                for index in range(self.no):
                    toReturnAO[index,u] = colFin[index]
            
            # Eq 39 solves active orbital equations of motion
            for q in range(self._scf.ncas):
                qMO = q + self._scf.ncore
                qOrth = self.mo_unitvec_to_orth(qMO)

                # First part of eq 28
                virt = np.matmul(self._h1e_orth.T,qOrth)

                # Eq 30, with dInv terms included
                for j in range(self._scf.ncas):
                    jMO = j + self._scf.ncore
                    for k in range(self._scf.ncas):
                        kMO = k+self._scf.ncore
                        for l in range(self._scf.ncas):
                            for m in range(self._scf.ncas):
                                mMO = m + self._scf.ncore
                                virt = virt + (self.casrdm2[l][m][j][k]*dInv[l,q]*np.matmul(self.get_w_orth(jMO,kMO),self.mo_unitvec_to_orth(mMO)))
                
                # Second part of eq 28. Note all core orbitals are dynamic core
                for u in range(self._scf.ncore):
                    virt = virt + (2*np.matmul(self.get_w_orth(u,u),qOrth)) - np.matmul(self.get_w_orth(u,qMO),self.mo_unitvec_to_orth(u))
                
                # Project into virtual space
                aoCol = np.matmul(qMat,virt)
                aoCol = aoCol + aoColForAct[q]

                '''
                # Now add core contributions
                for u in range(self._scf.ncore):
                    aoCol = aoCol + (toReturn[u,qMO]*self.mo_unitvec_to_orth(u))
                '''
                
                # Rotate to AO basis
                colFin = np.matmul(self.orth,aoCol)

                # Assign to output
                for index in range(self.no):
                    toReturnAO[index,qMO] = colFin[index]

            return toReturn, toReturnAO
    
    # Returns active space constant terms and 1e and 2e Hamiltonians
    # Analogous to final result on page 12 of group's dmet/dmet_original_jctc notes (cannot find equation in J. Chem. Theory Comput. 2013, 9, 3, 1428–1432)
    def get_actH(self,xMat):
        h1 = self._h1e_mo - xMat
        Econst = 2 * np.sum(np.diag(h1))
        for e in range(self._scf.ncore):
            for f in range(self._scf.ncore):
                Econst = Econst + (2*self._h2e_mo[e][e][f][f]) - self._h2e_mo[e][f][f][e]
        h1Mat = h1[self._scf.ncore:self.numP,self._scf.ncore:self.numP]
        for a in range(self._scf.ncas):
            for b in range(self._scf.ncas):
                for e in range(self._scf.ncore):
                    h1Mat[a,b] = h1Mat[a,b] + (2*self._h2e_mo[b+self._scf.ncore][a+self._scf.ncore][e][e]) - self._h2e_mo[b+self._scf.ncore][e][e][a+self._scf.ncore]
        h2Mat = self._h2e_mo[self._scf.ncore:self.numP,self._scf.ncore:self.numP,self._scf.ncore:self.numP,self._scf.ncore:self.numP]
        #return Econst,h1Mat,h2Mat
        return h1Mat,h2Mat

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
