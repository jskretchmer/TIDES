import numpy as np
from pyscf.tools import molden
from pyscf import gto, scf
from scipy.linalg import inv

'''
Basis Utility Functions
'''

def print2molden(mf, filename=None, mo_coeff=None):
    if mo_coeff is None: mo_coeff = mf.mo_coeff
    if filename is None: filename = 'file'
    with open(filename + '.molden', 'w') as moldenfile:
        molden.header(mf.mol, moldenfile)
        if len(np.shape(mo_coeff)) == 2:
            molden.orbital_coeff(mf.mol, moldenfile, mo_coeff,
                                 ene=mf.mo_energy, occ=mf.mo_occ)
        else:
            molden.orbital_coeff(mf.mol, moldenfile, mo_coeff[0],
                                 ene=mf.mo_energy[0], occ=mf.mo_occ[0],
                                 spin = 'Alpha')
            molden.orbital_coeff(mf.mol, moldenfile, mo_coeff[1],
                                 ene=mf.mo_energy[1], occ=mf.mo_occ[1],
                                 spin = 'Beta')

def match_fragment_atom(mf, frag):
    match_indices = [[] for idx, _ in enumerate(frag.mol._atom)]
    for idx, label in enumerate(mf.mol._atom):
        if label in frag.mol._atom:
            match_indices[frag.mol._atom.index(label)].append(idx)
    match_indices = [idx for idxs in match_indices for idx in idxs]
    return match_indices

def match_fragment_basis(mf, match_indices):
    match_basis = [[] for idx, _ in enumerate(match_indices)]
    for idx, bf in enumerate(mf.mol.ao_labels()):
        if int(bf.split()[0]) in match_indices:
            match_basis[match_indices.index(int(bf.split()[0]))].append(idx)
    match_basis = [b for bs in match_basis for b in bs]
    if mf.istype('GHF') | mf.istype('GKS'):
        # Account for bb and ab/ba blocks of generalized density matrix
        match_basis = np.concatenate((match_basis, [idx + mf.mol.nao for idx in match_basis]))
    
    return tuple(match_basis)
    
def mask_fragment_basis(mf, match_indices):
    match_basis = match_fragment_basis(mf, match_indices)

    if len(np.shape(mf.mo_coeff)) == 2:
        return np.ix_(match_basis, match_basis)
    else:
        return np.ix_((0,1),match_basis, match_basis)

def noscfbasis(mf, *fragments, reorder=True, orth=None):
    total_dim = np.shape(mf.mo_coeff)
    noscf_orbitals = np.zeros(total_dim)
    
    for frag in fragments:
        match_indices = match_fragment_atom(mf, frag)
        mask = mask_fragment_basis(mf, match_indices)
        noscf_orbitals[mask] = frag.mo_coeff
    
    if reorder:
        # Reorder so that all occupied orbitals appear before virtual orbitals
        # This may give the wrong order if you are projecting onto bizarre fragments, since the occupation of each fragment is used to reorder
        noscf_orbitals = reorder_noscf(noscf_orbitals, mf, *fragments)
    
    # Orthogonalize noscf orbitals (do this in orthogonal AO basis)
    if orth is None:
        orth = scf.addons.canonical_orth_(mf.get_ovlp())
    noscf_orth = np.matmul(inv(orth), noscf_orbitals).astype(np.complex128)
    noscf_orth, _ = np.linalg.qr(noscf_orth, 'complete')
    return np.matmul(orth, noscf_orth)


def reorder_noscf(noscf_orbitals, mf, *fragments):
    if len(np.shape(mf.mo_coeff)) == 3:
        occ_frag_a, occ_frag_b = [], []
        for frag in fragments:
            occ_frag_a.extend(frag.get_occ()[0])
            occ_frag_b.extend(frag.get_occ()[1])
        nind_a = occ_sort(occ_frag_a)
        nind_b = occ_sort(occ_frag_b)
        orbitals_a = noscf_orbitals[0,:,:]
        orbitals_b = noscf_orbitals[1,:,:]
        orbitals_a = orbitals_a[:,nind_a]
        orbitals_b = orbitals_b[:,nind_b]
        noscf_orbitals = np.stack((orbitals_a, orbitals_b))
    elif mf.istype('GHF') | mf.istype('GKS'):
        occ_frag = []
        match_basis = []
        for frag in fragments:
            occ_frag.extend(frag.get_occ())
            match_indices = match_fragment_atom(mf, frag)
            match_basis.extend(match_fragment_basis(mf, match_indices))

        occ_frag = np.array(occ_frag)[match_basis]
        nind = occ_sort(occ_frag)
        noscf_orbitals = noscf_orbitals[:,nind]
    else:
        occ_frag = []
        for frag in fragments:
            occ_frag.extend(frag.get_occ())
        nind = occ_sort(occ_frag)
        noscf_orbitals = noscf_orbitals[:,nind]
    return noscf_orbitals

def occ_sort(occ_list):
    nocc = []
    nvirt = []
    for idx, occ in enumerate(occ_list):
        if occ > 0:
            nocc.append(idx)
        else:
            nvirt.append(idx)
    return tuple(nocc + nvirt)

def read_mol(mol):
    _atom = mol._atom
    basis = mol.basis
    labels = [_atom[idx][0] for idx in range(len(_atom))]
    pos = [mol._atom[idx][1] for idx in range(len(_atom))]
    return basis, labels, pos
 
def write_mol(basis, labels, pos, spin=0, charge=0, unit='Bohr'):
    atom_str = '\n '
    for idx, R in enumerate(pos):
        atom_str += '\t'.join([labels[idx], str(R[0]), str(R[1]), str(R[2])]) + '\n '
    mol = gto.Mole(atom=atom_str, unit=unit, spin=spin, charge=charge)
    mol.basis = basis
    mol.build(verbose=0)
    return mol

