# PYTHON CODE FROM PYSCF FOR APPLYING THE HAMILTONIAN TO A FCI
# VECTOR WITHOUT ASSUMING ANY SYMMETRY OF THE HAMILTONIAN
# ONLY KEPT NECESSARY SUBROUTINES:
# ORIGNAL FILE OBTAINED FROM PYSCF FILE direct_spin0_o0.py
# contract_2e_complex WAS OBTAINED FROM PYSCF FILE direct_spin0_o0.py,
# REMOVING CALLS TO pyscf.ao2mo.restore
# absorb_1e WAS OBTAINED FROM PYSCF FILE direct_spin1.py,
# REMOVING CALLS TO pyscf.ao2mo.restore

import numpy
import numba
import pyscf.fci
from pyscf.fci import cistring

import time

#####################################################################


@numba.njit(cache=True)
def _scatter_alpha(link_indexa, fcivec, t1):
    na, ntab = link_indexa.shape[0], link_indexa.shape[1]
    nb = fcivec.shape[1]
    for str0 in range(na):
        for e in range(ntab):
            a = link_indexa[str0, e, 0]
            i = link_indexa[str0, e, 1]
            str1 = link_indexa[str0, e, 2]
            sign = link_indexa[str0, e, 3]
            for k in range(nb):
                t1[a, i, str1, k] += sign * fcivec[str0, k]


@numba.njit(cache=True)
def _scatter_beta(link_indexb, fcivec, t1):
    nb, ntab = link_indexb.shape[0], link_indexb.shape[1]
    na = fcivec.shape[0]
    for str0 in range(nb):
        for e in range(ntab):
            a = link_indexb[str0, e, 0]
            i = link_indexb[str0, e, 1]
            str1 = link_indexb[str0, e, 2]
            sign = link_indexb[str0, e, 3]
            for k in range(na):
                t1[a, i, k, str1] += sign * fcivec[k, str0]


@numba.njit(cache=True)
def _gather_alpha(link_indexa, t1, ci1):
    na, ntab = link_indexa.shape[0], link_indexa.shape[1]
    nb = ci1.shape[1]
    for str0 in range(na):
        for e in range(ntab):
            a = link_indexa[str0, e, 0]
            i = link_indexa[str0, e, 1]
            str1 = link_indexa[str0, e, 2]
            sign = link_indexa[str0, e, 3]
            for k in range(nb):
                ci1[str0, k] += sign * t1[i, a, str1, k]


@numba.njit(cache=True)
def _gather_beta(link_indexb, t1, ci1):
    nb, ntab = link_indexb.shape[0], link_indexb.shape[1]
    na = ci1.shape[0]
    for str0 in range(nb):
        for e in range(ntab):
            a = link_indexb[str0, e, 0]
            i = link_indexb[str0, e, 1]
            str1 = link_indexb[str0, e, 2]
            sign = link_indexb[str0, e, 3]
            for k in range(na):
                ci1[k, str0] += sign * t1[i, a, k, str1]


@numba.njit(cache=True)
def _scatter_alpha_ras(link_indexa, fcivec, alpha_level, allowed_beta_flat, allowed_beta_offset, t1):
    na, ntab = link_indexa.shape[0], link_indexa.shape[1]
    for str0 in range(na):
        for e in range(ntab):
            a = link_indexa[str0, e, 0]
            i = link_indexa[str0, e, 1]
            str1 = link_indexa[str0, e, 2]
            sign = link_indexa[str0, e, 3]
            LA = alpha_level[str1]
            lo = allowed_beta_offset[LA]
            hi = allowed_beta_offset[LA + 1]
            for idx in range(lo, hi):
                k = allowed_beta_flat[idx]
                t1[a, i, str1, k] += sign * fcivec[str0, k]


@numba.njit(cache=True)
def _scatter_beta_ras(link_indexb, fcivec, beta_level, allowed_alpha_flat, allowed_alpha_offset, t1):
    nb, ntab = link_indexb.shape[0], link_indexb.shape[1]
    for str0 in range(nb):
        for e in range(ntab):
            a = link_indexb[str0, e, 0]
            i = link_indexb[str0, e, 1]
            str1 = link_indexb[str0, e, 2]
            sign = link_indexb[str0, e, 3]
            LB = beta_level[str1]
            lo = allowed_alpha_offset[LB]
            hi = allowed_alpha_offset[LB + 1]
            for idx in range(lo, hi):
                k = allowed_alpha_flat[idx]
                t1[a, i, k, str1] += sign * fcivec[k, str0]


@numba.njit(cache=True)
def _gather_alpha_ras(link_indexa, t1, alpha_level, allowed_beta_flat, allowed_beta_offset, ci1):
    na, ntab = link_indexa.shape[0], link_indexa.shape[1]
    for str0 in range(na):
        LA = alpha_level[str0]
        lo = allowed_beta_offset[LA]
        hi = allowed_beta_offset[LA + 1]
        for e in range(ntab):
            a = link_indexa[str0, e, 0]
            i = link_indexa[str0, e, 1]
            str1 = link_indexa[str0, e, 2]
            sign = link_indexa[str0, e, 3]
            for idx in range(lo, hi):
                k = allowed_beta_flat[idx]
                ci1[str0, k] += sign * t1[i, a, str1, k]


@numba.njit(cache=True)
def _gather_beta_ras(link_indexb, t1, beta_level, allowed_alpha_flat, allowed_alpha_offset, ci1):
    nb, ntab = link_indexb.shape[0], link_indexb.shape[1]
    for str0 in range(nb):
        LB = beta_level[str0]
        lo = allowed_alpha_offset[LB]
        hi = allowed_alpha_offset[LB + 1]
        for e in range(ntab):
            a = link_indexb[str0, e, 0]
            i = link_indexb[str0, e, 1]
            str1 = link_indexb[str0, e, 2]
            sign = link_indexb[str0, e, 3]
            for idx in range(lo, hi):
                k = allowed_alpha_flat[idx]
                ci1[k, str0] += sign * t1[i, a, k, str1]


#####################################################################


def apply_ham_pyscf_check(
    CIcoeffs, hmat, Vmat, nalpha, nbeta, norbs, Econst, gen=False, fctr=0.5, link_index=None
):
    """
    subroutine that checks if the hamiltonian is real or complex
    and then calls the appropriate subroutine to apply the
    hamiltonian to a vector of CI coefficients using pyscf
    """

    if not gen:
        if numpy.iscomplexobj(hmat) and numpy.iscomplexobj(Vmat):
            # Complex restricted hamiltonian
            CIcoeffs = apply_ham_pyscf_complex(
                CIcoeffs, hmat, Vmat, nalpha, nbeta, norbs, Econst, fctr, link_index=link_index
            )

        elif not numpy.iscomplexobj(hmat) and not numpy.iscomplexobj(Vmat):
            # Real restricted hamiltonian
            CIcoeffs = apply_ham_pyscf_real(
                CIcoeffs, hmat, Vmat, nalpha, nbeta, norbs, Econst, fctr
            )

        else:
            print("ERROR: the 1e- integrals and 2e- integrals in")
            print("applyham_pyscf.apply_ham_pyscf_check")
            print("are NOT both real nor both complex")
            exit()

    elif gen:
        if numpy.iscomplexobj(hmat) and numpy.iscomplexobj(Vmat):
            # Complex generalized hamiltonian
            CIcoeffs = apply_ham_pyscf_spinor(
                CIcoeffs, hmat, Vmat, (nalpha + nbeta), norbs, Econst, fctr
            )

        elif not numpy.iscomplexobj(hmat) and not numpy.iscomplexobj(Vmat):
            # Real generalized hamiltonian
            CIcoeffs = apply_ham_pyscf_spinor(
                CIcoeffs, hmat, Vmat, (nalpha + nbeta), norbs, Econst, fctr
            )

        else:
            print("ERROR: the 1e- integrals and 2e- integrals in")
            print("applyham_pyscf.apply_ham_pyscf_check")
            print("are NOT both real nor both complex")
            exit()

    return CIcoeffs


#####################################################################


def apply_ham_pyscf_spinor(CIcoeffs, hmat, Vmat, nelec, norbs, Econst, fctr=0.5):
    """
    NOTE: This subroutine calls the PySCF fci solver for DHF or GHF,
     which can handle a complex Hamiltonian in a spinor basis. However,
     both hmat and Vmat must be either both complex or both real. This
     subroutine calls PySCF to apply a hamiltonian to a vector
     of CI coefficients.
     CIcoeffs is a 1d-array containing the CI coefficients in which
     the rows are coefficients for each configuration/determinant and
     the strings are ordered in ascending binary order with
     a 0/1 implies that a SPINOR orbital is empty/occupied.
     Vmat is the 2e- integrals and are given in chemistry notation.
     Econst is a constant energy contribution to the Hamiltonian.
     fctr is the factor in front of the 2e- terms
     when defining the hamiltonian; because this is a spinor basis,
     this is set to 1.0.
     NOTE: Currently only accepts hamiltonians and norbs in dimensionality
     of the spin-generalized formalism.
    """

    Vmat = pyscf.fci.fci_dhf_slow.absorb_h1e(hmat, Vmat, norbs, nelec, fctr)

    temp = pyscf.fci.fci_dhf_slow.contract_2e(Vmat, CIcoeffs, norbs, nelec)

    CIcoeffs = temp + Econst * CIcoeffs

    return CIcoeffs


#####################################################################


def apply_ham_pyscf_fully_complex(
    CIcoeffs, hmat, Vmat, nalpha, nbeta, norbs, Econst, fctr=0.5
):
    """
    subroutine that uses the apply_ham_pyscf_nosym
    subroutine below to apply a complex hamiltonian
    to a complex set of CI coefficients -
    also works if some subset are real, it's just slower
    """

    CIcoeffs = (
        apply_ham_pyscf_nosym(
            numpy.copy(CIcoeffs.real),
            numpy.copy(hmat.real),
            numpy.copy(Vmat.real),
            nalpha,
            nbeta,
            norbs,
            Econst,
            fctr,
        )
        - apply_ham_pyscf_nosym(
            numpy.copy(CIcoeffs.imag),
            numpy.copy(hmat.imag),
            numpy.copy(Vmat.imag),
            nalpha,
            nbeta,
            norbs,
            0.0,
            fctr,
        )
        + 1j
        * (
            apply_ham_pyscf_nosym(
                numpy.copy(CIcoeffs.imag),
                numpy.copy(hmat.real),
                numpy.copy(Vmat.real),
                nalpha,
                nbeta,
                norbs,
                Econst,
                fctr,
            )
            + apply_ham_pyscf_nosym(
                numpy.copy(CIcoeffs.real),
                numpy.copy(hmat.imag),
                numpy.copy(Vmat.imag),
                nalpha,
                nbeta,
                norbs,
                0.0,
                fctr,
            )
        )
    )

    return CIcoeffs


#####################################################################


def apply_ham_pyscf_real(CIcoeffs, hmat, Vmat, nalpha, nbeta, norbs, Econst, fctr=0.5):
    """
    NOTE: THIS SUBROUTINE ASSUMES THAT THE
     HAMILTONIAN IS SYMMETRIC (HERMITIAN AND REAL)
     AND IS CALLING PYSCF TO APPLY THE HAMILTONIAN
     subroutine to apply a hamiltonian to a vector
     of CI coefficients using pyscf
     CIcoeffs is a 2d-array containing the CI coefficients,
     the rows/columns correspond to the alpha/beta strings
     the strings are ordered in asscending binary order with
     a 0/1 implies that an orbital is empty/occupied
     the 2e- integrals, Vmat, are given in chemistry notation
     Econst is a constant energy contribution to the hamiltonian
     fctr is the factor in front of the 2e- terms
     when defining the hamiltonian
    """

    Vmat = pyscf.fci.direct_spin1.absorb_h1e(hmat, Vmat, norbs, (nalpha, nbeta), fctr)
    temp = pyscf.fci.direct_spin1.contract_2e(Vmat, CIcoeffs, norbs, (nalpha, nbeta))
    CIcoeffs = temp + Econst * CIcoeffs

    return CIcoeffs


#####################################################################


def apply_ham_pyscf_nosym(CIcoeffs, hmat, Vmat, nalpha, nbeta, norbs, Econst, fctr=0.5):
    """
    NOTE: THIS SUBROUTINE MAKES NO ASSUMPTION ABOUT THE SYMMETRY OF
    THE HAMILTONIAN, BUT CI COEFFICIENTS AND HAMILTONIAN MUST BE REAL
    AND IS CALLING PYSCF TO APPLY THE HAMILTONIAN
    subroutine to apply a hamiltonian to a vector of
    CI coefficients using pyscf
    CIcoeffs is a 2d-array containing the CI coefficients,
    the rows/columns correspond to the alpha/beta strings
    the strings are ordered in asscending binary order with
    a 0/1 implies that an orbital is empty/occupied
    the 2e- integrals, Vmat, are given in chemistry notation
    Econst is a constant energy contribution to the hamiltonian
    fctr is the factor in front of the 2e- terms when defining the hamiltonian
    """
    # print(f"CIcoeffs from beginning of apply_ham_nosym: \n {CIcoeffs}")
    Vmat_new = pyscf.fci.direct_nosym.absorb_h1e(
        hmat, Vmat, norbs, (nalpha, nbeta), fctr
    )
    temp = pyscf.fci.direct_nosym.contract_2e(
        Vmat_new, CIcoeffs, norbs, (nalpha, nbeta)
    )
    CIcoeffs_new = temp + Econst * CIcoeffs

    # print(f"V from direct_nosym: \n {Vmat_new}")
    # print(f"temp from contract_2e: \n {temp}")
    # print(f"CIcoeffs from nosym: \n {CIcoeffs_new}")
    # print(f"CI coefficients after applyhams: \n {CIcoeffs_new}")
    # nelec = nalpha + nbeta

    # CIgeneral = apply_ham_pyscf_spinor(
    #    CIcoeffs, hmat, Vmat, nelec, norbs, Econst, fctr=1.0
    # )
    # print(f"CI coefficients after general applyhams: \n {CIgeneral}")
    # print("ending at apply_ham_pyscf_nosym")
    # exit()
    return CIcoeffs_new


#####################################################################


def apply_ham_pyscf_complex(
    CIcoeffs, hmat, Vmat, nalpha, nbeta, norbs, Econst, fctr=0.5, link_index=None
):
    """
    NOTE: THIS SUBROUTINE ALLOWS FOR COMPLEX HAMILTONIAN,
    BUT ONLY REAL CI COEFFICIENTS
    AND IS USING THE SUBROUTINES IN THIS MODULE TO APPLY THE HAMILTONIAN
    subroutine to apply a hamiltonian to a vector
    of CI coefficients using pyscf
    CIcoeffs is a 2d-array containing the CI coefficients,
    the rows/columns correspond to the alpha/beta strings
    the strings are ordered in asscending binary order with a
    0/1 implies that an orbital is empty/occupied
    the 2e- integrals, Vmat, are given in chemistry notation
    Econst is a constant energy contribution to the hamiltonian
    fctr is the factor in front of the 2e- terms when defining the hamiltonian
    """
    Vmat = absorb_h1e_complex(hmat, Vmat, norbs, (nalpha, nbeta), fctr)
    temp = contract_2e_complex(Vmat, CIcoeffs, norbs, (nalpha, nbeta), link_index=link_index)
    CIcoeffs = temp + Econst * CIcoeffs

    return CIcoeffs


#####################################################################


def apply_ham_pyscf_complex_ras(
    CIcoeffs, hmat, Vmat, nalpha, nbeta, norbs, Econst, ras_blocks, fctr=0.5, link_index=None
):
    """
    NOTE: THIS SUBROUTINE ALLOWS FOR COMPLEX HAMILTONIAN,
    BUT ONLY REAL CI COEFFICIENTS
    AND IS USING THE SUBROUTINES IN THIS MODULE TO APPLY THE HAMILTONIAN
    subroutine to apply a hamiltonian to a vector
    of CI coefficients using pyscf
    CIcoeffs is a 2d-array containing the CI coefficients,
    the rows/columns correspond to the alpha/beta strings
    the strings are ordered in asscending binary order with a
    0/1 implies that an orbital is empty/occupied
    the 2e- integrals, Vmat, are given in chemistry notation
    Econst is a constant energy contribution to the hamiltonian
    ras_blocks is the (alpha_level, beta_level, allowed_beta_flat, allowed_beta_offset,
    allowed_alpha_flat, allowed_alpha_offset) tuple from rt_cas_ras's RAS setup
    fctr is the factor in front of the 2e- terms when defining the hamiltonian
    """
    Vmat = absorb_h1e_complex(hmat, Vmat, norbs, (nalpha, nbeta), fctr)
    temp = contract_2e_complex_ras(Vmat, CIcoeffs, norbs, (nalpha, nbeta), ras_blocks, link_index=link_index)
    CIcoeffs = temp + Econst * CIcoeffs

    return CIcoeffs


#####################################################################


def apply_ham_pyscf_complex_multi(
    CIcoeffs_list, hmat, Vmat, nalpha, nbeta, norbs, Econst, fctr=0.5, link_index=None
):
    """
    Applies the same complex Hamiltonian (hmat, Vmat) to several CI vectors
    that share it (e.g. the real and imaginary parts of a CI vector within
    one RK stage), absorbing the 1e Hamiltonian into the 2e integrals only
    once instead of once per CI vector.
    Returns a list of results in the same order as CIcoeffs_list.
    """
    Vmat_absorbed = absorb_h1e_complex(hmat, Vmat, norbs, (nalpha, nbeta), fctr)
    return [
        contract_2e_complex(Vmat_absorbed, CIcoeffs, norbs, (nalpha, nbeta), link_index=link_index)
        + Econst * CIcoeffs
        for CIcoeffs in CIcoeffs_list
    ]


#####################################################################


def apply_ham_pyscf_complex_ras_multi(
    CIcoeffs_list, hmat, Vmat, nalpha, nbeta, norbs, Econst, ras_blocks, fctr=0.5, link_index=None
):
    """
    RAS-masked counterpart to apply_ham_pyscf_complex_multi: applies the same
    complex Hamiltonian (hmat, Vmat) to several CI vectors that share it,
    absorbing the 1e Hamiltonian into the 2e integrals only once.
    Returns a list of results in the same order as CIcoeffs_list.
    """
    Vmat_absorbed = absorb_h1e_complex(hmat, Vmat, norbs, (nalpha, nbeta), fctr)
    return [
        contract_2e_complex_ras(Vmat_absorbed, CIcoeffs, norbs, (nalpha, nbeta), ras_blocks, link_index=link_index)
        + Econst * CIcoeffs
        for CIcoeffs in CIcoeffs_list
    ]


#####################################################################


def contract_2e_complex_ras(g2e, fcivec, norb, nelec, ras_blocks, link_index=None):
    """
    version of the pyscf subroutine contract_2e
    which allows for complex orbitals
    still assumes real CI coefficients
    removed calls to pyscf.ao2mo.restore
    other changes from pyscf have been noted
    subroutine follows logic of
    eqs 11.8.13-11.8.15 in helgaker, jorgensen and olsen

    ras_blocks is (alpha_level, beta_level, allowed_beta_flat, allowed_beta_offset,
    allowed_alpha_flat, allowed_alpha_offset), precomputed once in rt_cas_ras from
    the RAS excitation-level restriction. Since the restriction depends only on
    row_sums[i]+col_sums[j], every (levelA, levelB) block of the na x nb grid is
    either fully allowed or fully disallowed, so disallowed blocks are skipped
    entirely instead of being computed and then multiplied by a zero mask.
    """

    neleca, nelecb = nelec
    if link_index is None:
        link_indexa = cistring.gen_linkstr_index(range(norb), neleca)
        link_indexb = cistring.gen_linkstr_index(range(norb), nelecb)
    else:
        link_indexa, link_indexb = link_index

    (alpha_level, beta_level, allowed_beta_flat, allowed_beta_offset,
     allowed_alpha_flat, allowed_alpha_offset) = ras_blocks

    na = link_indexa.shape[0]
    nb = link_indexb.shape[0]
    fcivec = fcivec.reshape(na, nb)

    t1 = numpy.zeros((norb, norb, na, nb))
    _scatter_alpha_ras(link_indexa, fcivec, alpha_level, allowed_beta_flat, allowed_beta_offset, t1)
    _scatter_beta_ras(link_indexb, fcivec, beta_level, allowed_alpha_flat, allowed_alpha_offset, t1)

    # following line assumes the symmetry
    # that g[p,q,r,s]=g[r,s,p,q] in chemists notation
    # this symmetry holds for real and complex orbitals
    t1 = numpy.dot(g2e.reshape(norb * norb, -1), t1.reshape(norb * norb, -1))
    t1 = t1.reshape(norb, norb, na, nb)

    # data type of ci1 is now complex
    ci1 = numpy.zeros_like(fcivec, dtype=complex)
    _gather_alpha_ras(link_indexa, t1, alpha_level, allowed_beta_flat, allowed_beta_offset, ci1)
    _gather_beta_ras(link_indexb, t1, beta_level, allowed_alpha_flat, allowed_alpha_offset, ci1)
    return ci1


#####################################################################


def contract_2e_complex(g2e, fcivec, norb, nelec, link_index=None):
    """
    version of the pyscf subroutine contract_2e
    which allows for complex orbitals
    still assumes real CI coefficients
    removed calls to pyscf.ao2mo.restore
    other changes from pyscf have been noted
    subroutine follows logic of
    eqs 11.8.13-11.8.15 in helgaker, jorgensen and olsen
    """

    neleca, nelecb = nelec
    if link_index is None:
        link_indexa = cistring.gen_linkstr_index(range(norb), neleca)
        link_indexb = cistring.gen_linkstr_index(range(norb), nelecb)
    else:
        link_indexa, link_indexb = link_index

    na = link_indexa.shape[0]
    nb = link_indexb.shape[0]
    fcivec = fcivec.reshape(na, nb)

    t1 = numpy.zeros((norb, norb, na, nb))
    _scatter_alpha(link_indexa, fcivec, t1)
    _scatter_beta(link_indexb, fcivec, t1)

    # following line assumes the symmetry
    # that g[p,q,r,s]=g[r,s,p,q] in chemists notation
    # this symmetry holds for real and complex orbitals
    t1 = numpy.dot(g2e.reshape(norb * norb, -1), t1.reshape(norb * norb, -1))
    t1 = t1.reshape(norb, norb, na, nb)

    # data type of ci1 is now complex
    ci1 = numpy.zeros_like(fcivec, dtype=complex)
    _gather_alpha(link_indexa, t1, ci1)
    _gather_beta(link_indexb, t1, ci1)

    return ci1


#####################################################################


def absorb_h1e_complex(h1e, eri, norb, nelec, fac=1):
    # absorbing 1e- terms into 2e- terms
    # following 11.8.8 in helgaker, jorgensen and olsen
    # allows for no assumption about symmetry of 1 and 2 e- terms

    """Modify 2e Hamiltonian to include 1e Hamiltonian contribution."""
    if not isinstance(nelec, (int, numpy.integer)):
        nelec = sum(nelec)
    # eri = eri.copy()
    # h2e = pyscf.ao2mo.restore(1, eri, norb)
    h2e = eri.copy()
    f1e = h1e - numpy.einsum("jiik->jk", h2e) * 0.5
    f1e = f1e * (1.0 / nelec)
    for k in range(norb):
        h2e[k, k, :, :] += f1e
        h2e[:, :, k, k] += f1e

    return h2e * fac
