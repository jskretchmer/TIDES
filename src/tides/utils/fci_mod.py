# Mod that contains subroutines associated with pyscf FCI calculations

import numpy as np
from tides.utils import utils
from tides.utils import applyham_pyscf
import pyscf.fci
import scipy.linalg as la
from pyscf import gto, scf, ao2mo

#####################################################################
import time


def FCI_GS(h, V, Ecore, Norbs, Nele, casSize, gen=False):
    # Subroutine to perform groundstate FCI calculation using pyscf

    if isinstance(Nele, tuple):
        Nele = sum(Nele)

    # Define pyscf molecule
    mol = gto.M()
    mol.nelectron = Nele
    mol.nao = Norbs
    # this call is necessary to use user defined hamiltonian in fci step
    mol.incore_anyway = True
    if Nele // 2:
        mol.spin = 0
    else:
        mol.spin = 1

    if not gen:
        print(
            "WARNING: Currently GS FCI coefficients are not in final embedding basis."
        )

        # First perform restricted HF calculation
        mf = scf.RHF(mol)
        mf.get_hcore = lambda *args: h
        mf.get_ovlp = lambda *args: np.eye(Norbs)
        mf._eri = ao2mo.restore(8, V, Norbs)
        mf.kernel()
        '''
        mf.mo_coeff = np.eye(4)
        randomMat = np.random.randn(Norbs,Norbs)
        orthCo = la.orth(randomMat)
        print(np.allclose(np.eye(Norbs),orthCo@orthCo.T))
        mf.mo_coeff = np.copy(orthCo)
        '''

        # Perform FCI calculation using HF MOs
        #cisolver = pyscf.fci.FCI(mf, mf.mo_coeff)
        toReturn = pyscf.mcscf.CASCI(mf,casSize,casSize)
        toRun = toReturn.run()

        # NOTE: commented for use by TDFCI; if using this function for RT-pDMET,
        #       uncomment. Currently not used by RT-pDMET
        # E_FCI, CIcoeffs = pyscf.fci.direct_spin1.kernel(h, V, Norbs, Nele)

        #CIcoeffs = pyscf.fci.addons.transform_ci_for_orbital_rotation(
        #    CIcoeffs, Norbs, Nele, utils.adjoint(mf.mo_coeff)
        #)
        #CASCIcoeffs = pyscf.fci.addons.transform_ci_for_orbital_rotation(
        #    CASCIcoeffs, Norbs, Nele, utils.adjoint(mf.mo_coeff)
        #)

    if gen:
        # NOTE: HF det should be the most dominant det in FCI exp.,
        #       making num solver more stable than without.
        #       may not get correct answer for lattice models, so check
        #       for proper convergence.

        E_FCI, CIcoeffs = pyscf.fci.fci_dhf_slow.kernel(h, V, Norbs, Nele)

    return toRun.ci,mf.mo_coeff, toRun
    #E_CI, E_CAS, CASCIcoeffsNew, CASmo, CASmoE = toReturn.kernel(ci=CASCIcoeffs)
    #return mf,mf.mo_coeff,CASCIcoeffs.astype(np.complex128)
    #return CIcoeffs


#####################################################################


def get_corr1RDM(CIcoeffs, Norbs, Nele, gen=False):
    # Subroutine to get the FCI 1RDM
    # notation for restricted is dm_pq = < q^+ p >

    if not gen:
        if np.iscomplexobj(CIcoeffs):
            Re_CIcoeffs = np.copy(CIcoeffs.real)
            Im_CIcoeffs = np.copy(CIcoeffs.imag)

            corr1RDM = 1j * pyscf.fci.direct_spin1.trans_rdm1(
                Re_CIcoeffs, Im_CIcoeffs, Norbs, Nele
            )

            corr1RDM -= 1j * pyscf.fci.direct_spin1.trans_rdm1(
                Im_CIcoeffs, Re_CIcoeffs, Norbs, Nele
            )

            corr1RDM += pyscf.fci.direct_spin1.make_rdm1(Re_CIcoeffs, Norbs, Nele)

            corr1RDM += pyscf.fci.direct_spin1.make_rdm1(Im_CIcoeffs, Norbs, Nele)

        else:
            corr1RDM = pyscf.fci.direct_spin1.make_rdm1(CIcoeffs, Norbs, Nele)

    # Notation for generalized 1RDM from fci_dhf_slow is dm_pq = <|p^+ q|>
    # PySCF requires CIcoeffs to be in a spin-blocked configuration

    if gen:
        corr1RDM = pyscf.fci.fci_dhf_slow.make_rdm1(CIcoeffs, Norbs, Nele)

        if not np.allclose(np.diag(corr1RDM.imag), 0, atol=1e-9):
            print(
                "WARNING: NON-NEGLIGIBLE COMPLEX TERMS ALONG DIAGONAL OF EMBEDDED CORRELATED 1RDM"
            )
            print("-------- ENDING SIMULATION --------")
            exit()

        np.fill_diagonal(
            corr1RDM, corr1RDM.diagonal().real
        )  # make diagonal elements real

        # tranpose back to dm_pq = <|q^+ p|> to match restricted case
        corr1RDM = np.transpose(corr1RDM)

        if not la.ishermitian(corr1RDM, atol=1e-9):
            print("WARNING: EMBEDDED CORRELATED 1RDM IS NOT HERMITIAN")
            print("-------- ENDING SIMULATION --------")
            exit()

        corr1RDM = utils.make_hermitian(corr1RDM)

    return corr1RDM


#####################################################################


def get_corr12RDM(CIcoeffs, Norbs, Nele, gen=False):
    # Subroutine to get the FCI 1 & 2 RDMs together
    # Notation for restricted 1RDM is rho_pq = < c_q^dag c_p >
    # Notation for restricted 2RDM is gamma_prqs = < c_p^dag c_q^dag c_s c_r >

    if not gen:
        if np.iscomplexobj(CIcoeffs):
            Re_CIcoeffs = np.copy(CIcoeffs.real)
            Im_CIcoeffs = np.copy(CIcoeffs.imag)

            corr1RDM, corr2RDM = pyscf.fci.direct_spin1.trans_rdm12(
                Re_CIcoeffs, Im_CIcoeffs, Norbs, Nele
            )

            corr1RDM = corr1RDM * 1j
            corr2RDM = corr2RDM * 1j

            tmp1, tmp2 = pyscf.fci.direct_spin1.trans_rdm12(
                Im_CIcoeffs, Re_CIcoeffs, Norbs, Nele
            )

            corr1RDM -= 1j * tmp1
            corr2RDM -= 1j * tmp2

            tmp1, tmp2 = pyscf.fci.direct_spin1.make_rdm12(Re_CIcoeffs, Norbs, Nele)

            corr1RDM += tmp1
            corr2RDM += tmp2

            tmp1, tmp2 = pyscf.fci.direct_spin1.make_rdm12(Im_CIcoeffs, Norbs, Nele)

            corr1RDM += tmp1
            corr2RDM += tmp2

        else:

            corr1RDM, corr2RDM = pyscf.fci.direct_spin1.make_rdm12(
                CIcoeffs, Norbs, Nele
            )
            corr1RDM = pyscf.fci.direct_spin1.make_rdm1(CIcoeffs, Norbs, Nele)

    # Notation for generalized 1RDM is dm_pq = <|p^+ q|>
    # Notation for generalized 2RDM is dm_pq,rs = <|p^+ q r^+ s|>
    # This would be equivalent to (p_dag r_dag s q) in chemists notation, so equal to restricted notation
    # PySCF requires CIcoeffs to be in a spin-blocked configuration
    if gen:
        corr1RDM, corr2RDM = pyscf.fci.fci_dhf_slow.make_rdm12(CIcoeffs, Norbs, Nele)

        if not np.isclose(la.norm(CIcoeffs), 1.0, atol=1e-3):
            print(f"norm of CIcoeffs: {la.norm(CIcoeffs)}")

        if not np.allclose(np.diag(corr1RDM.imag), 0, atol=1e-9):
            print(la.norm(CIcoeffs))
            print(
                "WARNING: NON-NEGLIGIBLE COMPLEX TERMS ALONG DIAGONAL OF EMBEDDED CORRELATED 1RDM"
            )
            print("-------- ENDING SIMULATION --------")
            exit()

        np.fill_diagonal(
            corr1RDM, corr1RDM.diagonal().real
        )  # make diagonal elements real
        corr1RDM = np.transpose(corr1RDM)

        if not la.ishermitian(corr1RDM, atol=1e-9):
            print("WARNING: EMBEDDED CORRELATED 1RDM IS NOT HERMITIAN")
            print("-------- ENDING SIMULATION --------")
            exit()

        corr1RDM = utils.make_hermitian(corr1RDM)

    return corr1RDM, corr2RDM


#####################################################################


def get_trans1RDM(CIcoeffs_1, CIcoeffs_2, Norbs, Nele):
    # Subroutine to get the transition 1RDM between two CI vectors
    # notation is rho_pq = < psi_1 | c_q^dag c_p | psi_2 >

    if np.iscomplexobj(CIcoeffs_1) or np.iscomplexobj(CIcoeffs_2):
        Re_CIcoeffs_1 = np.copy(CIcoeffs_1.real)
        Im_CIcoeffs_1 = np.copy(CIcoeffs_1.imag)

        Re_CIcoeffs_2 = np.copy(CIcoeffs_2.real)
        Im_CIcoeffs_2 = np.copy(CIcoeffs_2.imag)

        corr1RDM = 1j * pyscf.fci.direct_spin1.trans_rdm1(
            Re_CIcoeffs_1, Im_CIcoeffs_2, Norbs, Nele
        )

        corr1RDM -= 1j * pyscf.fci.direct_spin1.trans_rdm1(
            Im_CIcoeffs_1, Re_CIcoeffs_2, Norbs, Nele
        )

        corr1RDM += pyscf.fci.direct_spin1.trans_rdm1(
            Re_CIcoeffs_1, Re_CIcoeffs_2, Norbs, Nele
        )

        corr1RDM += pyscf.fci.direct_spin1.trans_rdm1(
            Im_CIcoeffs_1, Im_CIcoeffs_2, Norbs, Nele
        )

    else:
        corr1RDM = pyscf.fci.direct_spin1.trans_rdm1(
            CIcoeffs_1, CIcoeffs_2, Norbs, Nele
        )

    return corr1RDM


#####################################################################


def get_FCI_E(h, V, Econst, CIcoeffs, Norbs, Nalpha, Nbeta, gen=False):
    # Subroutine to calculate the FCI electronic energy
    # for given Hamiltonian and FCI vector
    # Works with complex Hamitlonian and FCI vector

    if not gen:
        Hpsi = applyham_pyscf.apply_ham_pyscf_fully_complex(
            CIcoeffs, h, V, Nalpha, Nbeta, Norbs, Econst
        )

        Re_Hpsi = np.copy(Hpsi.real)
        Im_Hpsi = np.copy(Hpsi.imag)

        Re_CIcoeffs = np.copy(CIcoeffs.real)
        Im_CIcoeffs = np.copy(CIcoeffs.imag)

        FCI_E = pyscf.fci.addons.overlap(Re_CIcoeffs, Re_Hpsi, Norbs, (Nalpha, Nbeta))
        FCI_E += pyscf.fci.addons.overlap(Im_CIcoeffs, Im_Hpsi, Norbs, (Nalpha, Nbeta))
        FCI_E += 1j * pyscf.fci.addons.overlap(
            Re_CIcoeffs, Im_Hpsi, Norbs, (Nalpha, Nbeta)
        )
        FCI_E -= 1j * pyscf.fci.addons.overlap(
            Im_CIcoeffs, Re_Hpsi, Norbs, (Nalpha, Nbeta)
        )

    if gen:
        print("NOTE: FCI_E for generalized TDFCI still needs to be checked.")

        Nelec = Nalpha + Nbeta
        Hpsi = applyham_pyscf.apply_ham_pyscf_spinor(
            CIcoeffs, h, V, Nelec, Norbs, Econst
        )

        # NOTE: only difference between energy call and overlap procedure is in energy call ci1 is not
        # added to the original CIcoeffs while in Hpsi spinor

        Re_Hpsi = np.copy(Hpsi.real)
        Im_Hpsi = np.copy(Hpsi.imag)

        Re_CIcoeffs = np.copy(CIcoeffs.real)
        Im_CIcoeffs = np.copy(CIcoeffs.imag)

        FCI_E = pyscf.fci.addons.overlap(Re_CIcoeffs, Re_Hpsi, Norbs, Nelec)
        FCI_E += pyscf.fci.addons.overlap(Im_CIcoeffs, Im_Hpsi, Norbs, Nelec)
        FCI_E += 1j * pyscf.fci.addons.overlap(Re_CIcoeffs, Im_Hpsi, Norbs, Nelec)
        FCI_E -= 1j * pyscf.fci.addons.overlap(Im_CIcoeffs, Re_Hpsi, Norbs, Nelec)

        # did not match
        # taken from pyscf's energy call in fci_dhf_slow documentation
        h2e = pyscf.fci.fci_dhf_slow.absorb_h1e(h, V, Norbs, Nelec, 1.0)
        ci1 = pyscf.fci.fci_dhf_slow.contract_2e(h2e, CIcoeffs, Norbs, Nelec)
        FCI_E = np.dot(CIcoeffs.conj(), ci1)

        FCI_E, CI_coeffsnew = pyscf.fci.fci_dhf_slow.kernel(h, V, Norbs, Nelec)

    return FCI_E.real


#####################################################################
