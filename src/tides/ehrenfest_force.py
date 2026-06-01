import numpy
import copy
import ctypes
from pyscf import gto, scf, dft, grad
from pyscf import lib
from pyscf.lib import logger
from pyscf.scf import hf, _vhf
from pyscf.gto.mole import is_au
from pyscf.dft import numint
from scipy.linalg import fractional_matrix_power


def get_force(rt_ehrenfest):
    rt_ehrenfest._update_grad()

    v = rt_ehrenfest.evecs
    sqrt_e = numpy.sqrt(rt_ehrenfest.evals)
    etilde = 1. / (sqrt_e[:, numpy.newaxis] + sqrt_e[numpy.newaxis, :])
    Vinv = numpy.linalg.multi_dot([v, numpy.diag(1. / sqrt_e), v.T])

    grad_nuc = rt_ehrenfest._grad.grad_nuc()
    if rt_ehrenfest._scf.istype('RHF'):
        grad_elec = _grad_elec_restricted(rt_ehrenfest._grad, rt_ehrenfest.den_ao, etilde, v, Vinv)
    elif rt_ehrenfest._scf.istype('UHF'):
        grad_elec = _grad_elec_unrestricted(rt_ehrenfest._grad, rt_ehrenfest.den_ao, etilde, v, Vinv)
    elif rt_ehrenfest._scf.istype('GHF'):
        grad_elec = _grad_elec_generalized(rt_ehrenfest._grad, rt_ehrenfest.den_ao, etilde, v, Vinv)
    return -(grad_nuc + grad_elec)

def _grad_lowdin(mol):
    '''
    Lowdin/Symmetric Orthogonalization Matrix Gradient
    '''
    S = mol.intor('int1e_ovlp')
    Vinv = fractional_matrix_power(S, -0.5)
    e, v = numpy.linalg.eigh(S)
    sqrt_e = numpy.sqrt(e)
    etilde = 1. / (sqrt_e[:, numpy.newaxis] + sqrt_e[numpy.newaxis, :])
    return etilde, v, Vinv

def _grad_elec_restricted(scf_grad, den_ao=None, etilde=None, v=None, Vinv=None):
    '''
    Electronic part of RHF/RKS gradients for complex densities
    '''

    scf = scf_grad.base
    mol = scf_grad.mol
    mo_coeff = scf.mo_coeff
    mo_occ = scf.mo_occ

    if etilde is None or v is None or Vinv is None: etilde, v, Vinv = _grad_lowdin(mol)

    hcore_deriv = scf_grad.hcore_generator(mol)
    dS = scf_grad.get_ovlp()

    if den_ao is None: den_ao = scf.make_rdm1(mo_coeff, mo_occ)
    if den_ao.dtype != numpy.complex128:
        den_ao = den_ao.astype(numpy.complex128)

    fock_ao = scf.get_fock(dm=den_ao)

    vhf = scf_grad.get_veff(mol, den_ao)

    atmlst = range(mol.natm)
    aoslices = mol.aoslice_by_atom()
    de = numpy.zeros((len(atmlst),3))
    for k, ia in enumerate(atmlst):
        p0, p1 = aoslices[ia,2:]

        h1ao = hcore_deriv(ia)
        de[k] += numpy.einsum('xij,ji->x', h1ao, den_ao.real)

        de[k] += (numpy.einsum('xij,ji->x', vhf.real[:,p0:p1,:], den_ao.real[:,p0:p1]) - numpy.einsum('xij,ji->x', vhf.imag[:,p0:p1,:], den_ao.imag[:,p0:p1]))*2

        dStilde_bra = numpy.einsum('xik,kl->xil', numpy.einsum('ji,xjk->xik', v[p0:p1], dS[:,p0:p1]), v); dStilde_ket = numpy.einsum('xij->xji', dStilde_bra)
        dStilde = dStilde_bra + dStilde_ket
        dVtilde = numpy.einsum('ij,xij->xij', etilde, dStilde)
        dV = numpy.einsum('xik,lk->xil', numpy.einsum('ij,xjk->xik', v, dVtilde), v)
        VinvdV = numpy.einsum('ij,xjk->xik', Vinv, dV)
        PF = numpy.einsum('ij,jk->ik', den_ao, fock_ao)
        de[k] += -2 * numpy.einsum('xij,ji->x', VinvdV, PF).real

    if scf_grad.mol.symmetry:
        de = scf_grad.symmetrize(de, atmlst)
    if scf_grad.base.do_disp():
        de += scf_grad.get_dispersion()

    return de


def _grad_elec_unrestricted(scf_grad, den_ao=None, etilde=None, v=None, Vinv=None):
    '''
    Electronic part of UHF/UKS gradients for complex densities
    '''

    scf = scf_grad.base
    mol = scf_grad.mol
    mo_coeff = scf.mo_coeff
    mo_occ = scf.mo_occ

    if etilde is None or v is None or Vinv is None: etilde, v, Vinv = _grad_lowdin(mol)

    hcore_deriv = scf_grad.hcore_generator(mol)
    dS = scf_grad.get_ovlp()

    if den_ao is None: den_ao = scf.make_rdm1(mo_coeff, mo_occ)
    if den_ao.dtype != numpy.complex128:
        den_ao = den_ao.astype(numpy.complex128)

    fock_ao = scf.get_fock(dm=den_ao)

    vhf = scf_grad.get_veff(mol, den_ao)

    atmlst = range(mol.natm)
    aoslices = mol.aoslice_by_atom()
    de = numpy.zeros((len(atmlst),3))
    for k, ia in enumerate(atmlst):
        p0, p1 = aoslices[ia,2:]

        h1ao = hcore_deriv(ia)
        de[k] += numpy.einsum('xij,sji->x', h1ao, den_ao.real)

        de[k] += (numpy.einsum('sxij,sji->x', vhf.real[:,:,p0:p1,:], den_ao.real[:,:,p0:p1]) - numpy.einsum('sxij,sji->x', vhf.imag[:,:,p0:p1,:], den_ao.imag[:,:,p0:p1]))*2

        dStilde_bra = numpy.einsum('xik,kl->xil', numpy.einsum('ji,xjk->xik', v[p0:p1], dS[:,p0:p1]), v); dStilde_ket = numpy.einsum('xij->xji', dStilde_bra)
        dStilde = dStilde_bra + dStilde_ket
        dVtilde = numpy.einsum('ij,xij->xij', etilde, dStilde)
        dV = numpy.einsum('xik,lk->xil', numpy.einsum('ij,xjk->xik', v, dVtilde), v)
        VinvdV = numpy.einsum('ij,xjk->xik', Vinv, dV)
        PF = numpy.einsum('sij,sjk->ik', den_ao, fock_ao)
        de[k] += -2 * numpy.einsum('xij,ji->x', VinvdV, PF).real

    if scf_grad.mol.symmetry:
        de = scf_grad.symmetrize(de, atmlst)
    if scf_grad.base.do_disp():
        de += scf_grad.get_dispersion()

    return de


def _get_ovlp_generalized(mol): 
    ds = grad.rhf.get_ovlp(mol)
    return numpy.block([[ds, numpy.zeros_like(ds)], [numpy.zeros_like(ds), ds]])

def _hcore_generator_generalized(mf_grad, mol): 
    hcore_deriv = grad.rhf.hcore_generator(mf_grad, mol)
    def _generalized_hcore_deriv(atm_id):
        h1ao = hcore_deriv(atm_id)
        _generalized_h1ao = numpy.block([[h1ao, numpy.zeros_like(h1ao)], [numpy.zeros_like(h1ao), h1ao]])
        return _generalized_h1ao
    return _generalized_hcore_deriv

def _grad_elec_generalized(scf_grad, den_ao=None, etilde=None, v=None, Vinv=None):
    '''
    Electronic part of GHF/GKS gradients for complex densities
    '''

    scf = scf_grad.base
    mol = scf_grad.mol
    nao = mol.nao
    mo_coeff = scf.mo_coeff
    mo_occ = scf.mo_occ

    if etilde is None or v is None or Vinv is None: etilde, v, Vinv = _grad_lowdin(mol)

    hcore_deriv = _hcore_generator_generalized(scf_grad, mol)
    dS = _get_ovlp_generalized(mol)

    if den_ao is None: den_ao = scf.make_rdm1(mo_coeff, mo_occ)
    if den_ao.dtype != numpy.complex128:
        den_ao = den_ao.astype(numpy.complex128)

    fock_ao = scf.get_fock(dm=den_ao)

    vhf = scf_grad.get_veff(mol, den_ao)

    atmlst = range(mol.natm)
    aoslices = mol.aoslice_by_atom()
    de = numpy.zeros((len(atmlst),3))
    for k, ia in enumerate(atmlst):
        p0, p1 = aoslices[ia,2:]

        h1ao = hcore_deriv(ia)
        de[k] += numpy.einsum('xij,ji->x', h1ao, den_ao.real)

        de[k] += (numpy.einsum('xij,ji->x', vhf.real[:,p0:p1,:], den_ao.real[:,p0:p1]) - numpy.einsum('xij,ji->x', vhf.imag[:,p0:p1,:], den_ao.imag[:,p0:p1])) * 2
        de[k] += (numpy.einsum('xij,ji->x', vhf.real[:,p0+nao:p1+nao,:], den_ao.real[:,p0+nao:p1+nao]) - numpy.einsum('xij,ji->x', vhf.imag[:,p0+nao:p1+nao,:], den_ao.imag[:,p0+nao:p1+nao])) * 2

        dStilde_bra = numpy.einsum('xik,kl->xil', numpy.einsum('ji,xjk->xik', v[p0:p1], dS[:,p0:p1]), v)
        dStilde_bra += numpy.einsum('xik,kl->xil', numpy.einsum('ji,xjk->xik', v[p0+nao:p1+nao], dS[:,p0+nao:p1+nao]), v)
        dStilde_ket = numpy.einsum('xij->xji', dStilde_bra)
        dStilde = dStilde_bra + dStilde_ket
        dVtilde = numpy.einsum('ij,xij->xij', etilde, dStilde)
        dV = numpy.einsum('xik,lk->xil', numpy.einsum('ij,xjk->xik', v, dVtilde), v)
        VinvdV = numpy.einsum('ij,xjk->xik', Vinv, dV)
        PF = numpy.einsum('ij,jk->ik', den_ao, fock_ao)
        de[k] += -2 * numpy.einsum('xij,ji->x', VinvdV, PF).real

    if scf_grad.mol.symmetry:
        de = scf_grad.symmetrize(de, atmlst)
    if scf_grad.base.do_disp():
        de += scf_grad.get_dispersion()

    return de

