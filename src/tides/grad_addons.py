import numpy
from pyscf import lib
from pyscf.lib import logger
from pyscf import scf, dft, grad
from pyscf.grad import rks as rks_grad
from pyscf.grad import uks as uks_grad

# This is just a wrapper for the mf_grad objects that allows for complex 1rdm's
def complex_veff_(mf_grad):
    def get_veff_rks(ks_grad, mol=None, dm=None):
        if mol is None: mol = ks_grad.mol
        if dm is None: dm = ks_grad.base.make_rdm1()
        if dm.dtype != numpy.complex128:
            dm = dm.astype(numpy.complex128)
        t0 = (logger.process_clock(), logger.perf_counter())

        mf = ks_grad.base
        ni = mf._numint
        grids, nlcgrids = rks_grad._initialize_grids(ks_grad)

        mem_now = lib.current_memory()[0]
        max_memory = max(2000, ks_grad.max_memory*.9-mem_now)
        if ks_grad.grid_response:
            exc_re, vxc_re = rks_grad.get_vxc_full_response(ni, mol, grids, mf.xc, dm.real,
                                             hermi=1,
                                             max_memory=max_memory,
                                             verbose=ks_grad.verbose)
            exc_im, vxc_im = rks_grad.get_vxc_full_response(ni, mol, grids, mf.xc, dm.imag,
                                             hermi=0,
                                             max_memory=max_memory,
                                             verbose=ks_grad.verbose)
            exc = exc_re + 1j * exc_im
            vxc = vxc_re + 1j * vxc_im
            if mf.do_nlc():
                if ni.libxc.is_nlc(mf.xc):
                    xc = mf.xc
                else:
                    xc = mf.nlc
                enlc_re, vnlc_re = rks_grad.get_nlc_vxc_full_response(
                    ni, mol, nlcgrids, xc, dm.real,
                    hermi=1,
                    max_memory=max_memory, verbose=ks_grad.verbose)
                enlc_im, vnlc_im = rks_grad.get_nlc_vxc_full_response(
                    ni, mol, nlcgrids, xc, dm.imag,
                    hermi=0,
                    max_memory=max_memory, verbose=ks_grad.verbose)
                exc += enlc_re + 1j * enlc_im
                vxc += vnlc_re + 1j * vnlc_im
            logger.debug1(ks_grad, 'sum(grids response) %s', exc.sum(axis=0))
        else:
            exc_re, vxc_re = rks_grad.get_vxc(ni, mol, grids, mf.xc, dm.real,
                               hermi=1,
                               max_memory=max_memory, verbose=ks_grad.verbose)
            exc_im, vxc_im = rks_grad.get_vxc(ni, mol, grids, mf.xc, dm.imag,
                               hermi=0,
                               max_memory=max_memory, verbose=ks_grad.verbose)
            exc = None
            vxc = vxc_re + 1j * vxc_im
            if mf.do_nlc():
                if ni.libxc.is_nlc(mf.xc):
                    xc = mf.xc
                else:
                    xc = mf.nlc
                enlc_re, vnlc_re = rks_grad.get_nlc_vxc(
                    ni, mol, nlcgrids, xc, dm.real,
                    hermi=1,
                    max_memory=max_memory, verbose=ks_grad.verbose)
                enlc_im, vnlc_im = rks_grad.get_nlc_vxc(
                    ni, mol, nlcgrids, xc, dm.imag,
                    hermi=0,
                    max_memory=max_memory, verbose=ks_grad.verbose)
                vxc += vnlc_re + 1j * vnlc_im
        t0 = logger.timer(ks_grad, 'vxc', *t0)

        if not ni.libxc.is_hybrid_xc(mf.xc):
            vj_re = ks_grad.get_j(mol, dm.real)
            vj_im = ks_grad.get_j(mol, dm.imag)
            vj = vj_re + 1j * vj_im
            vxc += vj
        else:
            omega, alpha, hyb = ni.rsh_and_hybrid_coeff(mf.xc, spin=mol.spin)
            vj_re, vk_re = ks_grad.get_jk(mol, dm.real)
            vj_im, vk_im = ks_grad.get_jk(mol, dm.imag)
            vj = vj_re + 1j * vj_im
            vk = vk_re + 1j * vk_im
            vk *= hyb
            if omega != 0:
                vk += ks_grad.get_k(mol, dm.real, omega=omega) * (alpha - hyb)
                vk += 1j * ks_grad.get_k(mol, dm.imag, omega=omega) * (alpha - hyb)
            vxc += vj - vk * .5
        return lib.tag_array(vxc, exc1_grid=exc)

    def get_veff_rhf(hf_grad, mol=None, dm=None):
        if mol is None: mol = hf_grad.mol
        if dm is None: dm = hf_grad.base.make_rdm1()
        if dm.dtype != numpy.complex128:
            dm = dm.astype(numpy.complex128)

        vj_re, vk_re = hf_grad.get_jk(mol, dm.real)
        vj_im, vk_im = hf_grad.get_jk(mol, dm.imag)
        vj = vj_re + 1j * vj_im
        vk = vk_re + 1j * vk_im
        return vj - vk * .5

    def get_veff_uks(ks_grad, mol=None, dm=None):
        if mol is None: mol = ks_grad.mol
        if dm is None: dm = ks_grad.base.make_rdm1()
        if dm.dtype != numpy.complex128:
            dm = dm.astype(numpy.complex128)
        t0 = (logger.process_clock(), logger.perf_counter())

        mf = ks_grad.base
        ni = mf._numint
        grids, nlcgrids = rks_grad._initialize_grids(ks_grad)

        ni = mf._numint
        mem_now = lib.current_memory()[0]
        max_memory = max(2000, ks_grad.max_memory*.9-mem_now)
        if ks_grad.grid_response:
            exc_re, vxc_re = uks_grad.get_vxc_full_response(ni, mol, grids, mf.xc, dm.real,
                                             hermi=1,
                                             max_memory=max_memory,
                                             verbose=ks_grad.verbose)
            exc_im, vxc_im = uks_grad.get_vxc_full_response(ni, mol, grids, mf.xc, dm.imag,
                                             hermi=0,
                                             max_memory=max_memory,
                                             verbose=ks_grad.verbose)
            exc = exc_re + 1j * exc_im
            vxc = vxc_re + 1j * vxc_im
            if mf.do_nlc():
                if ni.libxc.is_nlc(mf.xc):
                    xc = mf.xc
                else:
                    xc = mf.nlc
                enlc_re, vnlc_re = rks_grad.get_nlc_vxc_full_response(
                    ni, mol, nlcgrids, xc, (dm[0]+dm[1]).real,
                    hermi=1,
                    max_memory=max_memory, verbose=ks_grad.verbose)
                enlc_im, vnlc_im = rks_grad.get_nlc_vxc_full_response(
                    ni, mol, nlcgrids, xc, (dm[0]+dm[1]).imag,
                    hermi=0,
                    max_memory=max_memory, verbose=ks_grad.verbose)
                exc += enlc_re + 1j * enlc_im
                vxc += vnlc_re + 1j * vnlc_im
            logger.debug1(ks_grad, 'sum(grids response) %s', exc.sum(axis=0))
        else:
            exc_re, vxc_re = uks_grad.get_vxc(ni, mol, grids, mf.xc, dm.real,
                               hermi=1,
                               max_memory=max_memory, verbose=ks_grad.verbose)
            exc_im, vxc_im = uks_grad.get_vxc(ni, mol, grids, mf.xc, dm.imag,
                               hermi=0,
                               max_memory=max_memory, verbose=ks_grad.verbose)
            exc = exc_re + 1j * exc_im
            vxc = vxc_re + 1j * vxc_im
            if mf.do_nlc():
                if ni.libxc.is_nlc(mf.xc):
                    xc = mf.xc
                else:
                    xc = mf.nlc
                enlc_re, vnlc_re = rks_grad.get_nlc_vxc(
                    ni, mol, nlcgrids, xc, (dm[0]+dm[1]).real,
                    hermi=1,
                    max_memory=max_memory, verbose=ks_grad.verbose)
                enlc_im, vnlc_im = rks_grad.get_nlc_vxc(
                    ni, mol, nlcgrids, xc, (dm[0]+dm[1]).imag,
                    hermi=0,
                    max_memory=max_memory, verbose=ks_grad.verbose)
                vxc += vnlc_re + 1j * vnlc_im
        t0 = logger.timer(ks_grad, 'vxc', *t0)

        if not ni.libxc.is_hybrid_xc(mf.xc):
            vj_re = ks_grad.get_j(mol, dm.real)
            vj_im = ks_grad.get_j(mol, dm.imag)
            vj = vj_re + 1j * vj_im
            vxc += vj[0] + vj[1]
        else:
            omega, alpha, hyb = ni.rsh_and_hybrid_coeff(mf.xc, spin=mol.spin)
            vj_re, vk_re = ks_grad.get_jk(mol, dm.real)
            vj_im, vk_im = ks_grad.get_jk(mol, dm.imag)
            vj = vj_re + 1j * vj_im
            vk = vk_re + 1j * vk_im
            vk *= hyb
            if omega != 0:
                vk += ks_grad.get_k(mol, dm.real, omega=omega) * (alpha - hyb)
                vk += 1j * ks_grad.get_k(mol, dm.imag, omega=omega) * (alpha - hyb)
            vxc += vj[0] + vj[1] - vk

        return lib.tag_array(vxc, exc1_grid=exc)

    def get_veff_uhf(hf_grad, mol=None, dm=None):
        if mol is None: mol = hf_grad.mol
        if dm is None: dm = hf_grad.base.make_rdm1()
        if dm.dtype != numpy.complex128:
            dm = dm.astype(numpy.complex128)

        vj_re, vk_re = hf_grad.get_jk(mol, dm.real)
        vj_im, vk_im = hf_grad.get_jk(mol, dm.imag)
        vj = vj_re + 1j * vj_im
        vk = vk_re + 1j * vk_im
        return vj[0]+vj[1] - vk

    def get_veff_gks(ks_grad, mol=None, dm=None):
        if mol is None: mol = ks_grad.mol
        if dm is None: dm = ks_grad.base.make_rdm1()
        if dm.dtype != numpy.complex128:
            dm = dm.astype(numpy.complex128)

        mf = ks_grad.base
        ni = mf._numint

        nao = mol.nao
        dmaa = dm[:nao,:nao]
        dmbb = dm[nao:,nao:]
        dmab = dm[:nao,nao:]
        dmba = dm[nao:,:nao]
        vxc = numpy.zeros((3, 2 * nao, 2 * nao), dtype=numpy.complex128)

        umf = dft.uks.UKS(mol)
        umf.xc = mf.xc
        if hasattr(ni, "omega"):
            umf._numint.omega = ni.omega
        if hasattr(ni, "alpha"):
            umf._numint.alpha = ni.alpha
        if hasattr(ni, "beta"):
            umf._numint.beta = ni.beta
        umf_grad = umf.apply(grad.UKS)

        umf_veff = get_veff_uks(umf_grad, mol, numpy.array([dmaa,dmbb]))
        umf_ni = umf_grad.base._numint
        if umf_ni.libxc.is_hybrid_xc(umf.xc):
            omega, alpha, hyb = umf_ni.rsh_and_hybrid_coeff(umf.xc, spin=mol.spin)
            vkab_re = umf_grad.get_k(mol, dmab.real) * hyb
            vkab_im = umf_grad.get_k(mol, dmab.imag) * hyb
            vkba_re = umf_grad.get_k(mol, dmba.real) * hyb
            vkba_im = umf_grad.get_k(mol, dmba.imag) * hyb
            vkab = vkab_re + 1j * vkab_im
            vkba = vkba_re + 1j * vkba_im
            if omega != 0:
                vkab += umf_grad.get_k(mol, dmab.real, omega=omega) * (alpha - hyb)
                vkab += 1j * umf_grad.get_k(mol, dmab.imag, omega=omega) * (alpha - hyb)
                vkba += umf_grad.get_k(mol, dmba.real, omega=omega) * (alpha - hyb)
                vkba += 1j * umf_grad.get_k(mol, dmba.imag, omega=omega) * (alpha - hyb)
            vxc[:,:nao,nao:] = -vkab
            vxc[:,nao:,:nao] = -vkba
        vxc[:,:nao,:nao] = umf_veff[0]
        vxc[:,nao:,nao:] = umf_veff[1]

        return vxc

    def get_veff_ghf(hf_grad, mol=None, dm=None):
        if mol is None: mol = hf_grad.mol
        if dm is None: dm = hf_grad.base.make_rdm1()
        if dm.dtype != numpy.complex128:
            dm = dm.astype(numpy.complex128)

        nao = mol.nao
        dmaa = dm[:nao,:nao]
        dmbb = dm[nao:,nao:]
        dmab = dm[:nao,nao:]
        dmba = dm[nao:,:nao]
        vxc = numpy.zeros((3, 2 * nao, 2 * nao), dtype=numpy.complex128)

        umf = scf.uhf.UHF(mol)
        umf_grad = umf.apply(grad.UHF)
        umf_veff = get_veff_uhf(umf_grad, mol, numpy.array([dmaa,dmbb]))
        vkab_re = umf_grad.get_k(mol, dmab.real)
        vkab_im = umf_grad.get_k(mol, dmab.imag)
        vkba_re = umf_grad.get_k(mol, dmba.real)
        vkba_im = umf_grad.get_k(mol, dmba.imag)
        vkab = vkab_re + 1j * vkab_im
        vkba = vkba_re + 1j * vkba_im

        vxc[:,:nao,nao:] = -vkab
        vxc[:,nao:,:nao] = -vkba
        vxc[:,:nao,:nao] = umf_veff[0]
        vxc[:,nao:,nao:] = umf_veff[1]
        return vxc

    if mf_grad.base.istype('RKS'):
        mf_grad.get_veff = lambda mol=None, dm=None, ks_grad=mf_grad: get_veff_rks(ks_grad, mol, dm)
    elif mf_grad.base.istype('RHF'):
        mf_grad.get_veff = lambda mol=None, dm=None, hf_grad=mf_grad: get_veff_rhf(hf_grad, mol, dm)
    elif mf_grad.base.istype('UKS'):
        mf_grad.get_veff = lambda mol=None, dm=None, ks_grad=mf_grad: get_veff_uks(ks_grad, mol, dm)
    elif mf_grad.base.istype('UHF'):
        mf_grad.get_veff = lambda mol=None, dm=None, hf_grad=mf_grad: get_veff_uhf(hf_grad, mol, dm)
    elif mf_grad.base.istype('GKS'):
        mf_grad.get_veff = lambda mol=None, dm=None, ks_grad=mf_grad: get_veff_gks(ks_grad, mol, dm)
    elif mf_grad.base.istype('GHF'):
        mf_grad.get_veff = lambda mol=None, dm=None, hf_grad=mf_grad: get_veff_ghf(hf_grad, mol, dm)
    else:
        print(f"Complex 1RDM gradients not implemented for {type(mf_grad.base)}")

    return mf_grad

