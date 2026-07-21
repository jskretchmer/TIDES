import sys
import numpy as np
from pyscf import gto, scf, dft, data
from pyscf.scf.atom_ks import get_atm_nrks
from pyscf.dft import numint

try:
    # pip install git+https://github.com/frobnitzem/hirshfeld
    from pyscf.hirshfeld import HirshfeldAnalysis
except ImportError:
    sys.stderr.write('Note: Hirshfeld module not installed. Install with [pip install git+https://github.com/frobnitzem/hirshfeld] if you wish to collect hirshfeld observables.')

def hirshfeld_partition(scf, den_ao, grids=None, atom_weights=None):
    if grids is None or atom_weights is None: grids, atom_weights = get_weights(scf.mol)
    den_ao = den_ao.astype(np.complex128)
    
    # Restricted
    if scf.istype('RHF') | scf.istype('RKS'):
        rho = _cast_den_ao(scf, den_ao, grids)
        return rho * atom_weights
    # Unrestricted - break into aa, bb spin blocks
    if scf.istype('UHF') | scf.istype('UKS'):
        rho_a = _cast_den_ao(scf, den_ao[0], grids)
        rho_b = _cast_den_ao(scf, den_ao[1], grids)
        return rho_a * atom_weights, rho_b * atom_weights
    # Generalized - break into aa, ab, ba, bb spin blocks
    if scf.istype('GHF') | scf.istype('GKS'):
        ovlp = scf.get_ovlp()
        Nsp = int(ovlp.shape[0]/2)

        rho_aa = _cast_den_ao(scf, den_ao[:Nsp, :Nsp], grids)
        rho_ab = _cast_den_ao(scf, den_ao[:Nsp, Nsp:], grids)
        rho_ba = _cast_den_ao(scf, den_ao[Nsp:, :Nsp], grids)
        rho_bb = _cast_den_ao(scf, den_ao[Nsp:, Nsp:], grids)
        return rho_aa * atom_weights, rho_ab * atom_weights, rho_ba * atom_weights, rho_bb * atom_weights

def _cast_den_ao(scf, den_ao, grids):
    # Re Part
    rho_re = numint.get_rho(numint.NumInt(), scf.mol, np.real(den_ao), grids)
    # Imag Part
    rho_imag = numint.get_rho(numint.NumInt(), scf.mol, np.imag(den_ao), grids)

    return (rho_re + 1j * rho_imag) * grids.weights

def get_weights(mol):
    grids = dft.Grids(mol)
    grids.build()
    scf = dft.RKS(mol)
    scf.verbose = 0
    scf.xc = 'HF'
    scf.grids = grids
    scf.kernel()

    H = HirshfeldAnalysis(scf).run()
    return grids, H.result['weights_free']
