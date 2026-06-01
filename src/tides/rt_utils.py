import numpy as np
from pyscf import scf
from tides.basis_utils import _match_fragment_atom, _mask_fragment_basis, noscfbasis, _read_mol, _write_mol
from tides import ehrenfest_force

'''
Real-time Utilities
'''

def excite(rt_scf, excitation_alpha=None, excitation_beta=None):
    # Excite an electron from the index specified
    if rt_scf.nmat == 1:
        excitation = excitation_alpha
        rt_scf.occ[excitation-1] -= 1
    else:
        if excitation_alpha:
            rt_scf.occ[0][excitation_alpha-1] -= 1
        if excitation_beta:
            rt_scf.occ[1][excitation_beta-1] -= 1

    rt_scf.den_ao = rt_scf._scf.make_rdm1(mo_occ=rt_scf.occ)

def input_fragments(rt_scf, *fragments):
    # Specify the relevant atom indices for each fragment
    # The charge, energy, dipole, and magnetization on each fragment can be calculated
    # Also useful for MO occ projections onto fragment MOs

    nmo = rt_scf._scf.mol.nao_nr()
    for idx, frag in enumerate(fragments):
        match_indices = _match_fragment_atom(rt_scf._scf, frag)
        mask_basis = _mask_fragment_basis(rt_scf._scf, match_indices)
        frag.match_indices = match_indices
        frag.mask = mask_basis
        rt_scf.fragments.append(frag)

def _update_mo_coeff_print(rt_ehrenfest):
    rt_ehrenfest.get_mo_coeff_print(rt_ehrenfest)

def get_scf_orbitals(rt_ehrenfest):
    mo_coeff = np.copy(rt_ehrenfest._scf.mo_coeff)
    rt_ehrenfest._scf.kernel()
    rt_ehrenfest.mo_coeff_print = rt_ehrenfest._scf.mo_coeff
    rt_ehrenfest._scf.mo_coeff = mo_coeff

def get_noscf_orbitals(rt_ehrenfest):
    # Update fragments to new geometry, solve scf problem
    basis, labels, pos = _read_mol(rt_ehrenfest._scf.mol)
    for frag in rt_ehrenfest.fragments:
        frag_indices = frag.match_indices
        frag_labels = [labels[i] for i in frag_indices]
        frag_pos = [pos[i] for i in frag_indices]
        frag.reset(_write_mol(basis, frag_labels, frag_pos, spin=frag.mol.spin, charge=frag.mol.charge))
        frag.verbose = 0
        frag.kernel()
        frag.match_indices = frag_indices
    rt_ehrenfest.mo_coeff_print = noscfbasis(rt_ehrenfest._scf, *rt_ehrenfest.fragments)


def restart_from_chkfile(rt_scf):
    rt_scf._log.note(f'### Restarting from chkfile: {rt_scf.chkfile} ###\n')
    with open(rt_scf.chkfile, 'r') as f:
        chk_lines = f.readlines()
        rt_scf.current_time = float(chk_lines[0].split()[3])
        if 'Position (AU): \n' in chk_lines:
            a1 = chk_lines.index('Position (AU): \n')
            a2 = chk_lines.index('Velocity (AU): \n')
            a3 = chk_lines.index('Force (AU): \n')
        else:
            a1 = None
        if rt_scf.nmat == 1:
            if rt_scf.prop == 'magnus_interpol':
                for idx, line in enumerate(chk_lines):
                    if 'Fock Orth N12 dt' in line:
                        b1 = idx
                        break
                rt_scf._scf.mo_coeff = np.loadtxt(chk_lines[2:b1], dtype=np.complex128)
                rt_scf._scf._fock_orth_n12dt = np.loadtxt(chk_lines[b1+1:a1], dtype=np.complex128)
            elif rt_scf.prop == 'magnus_step':
                for idx, line in enumerate(chk_lines):
                    if 'MO Coeff Orth Old' in line:
                        b1 = idx
                        break
                rt_scf._scf.mo_coeff = np.loadtxt(chk_lines[2:b1], dtype=np.complex128)
                rt_scf._scf.mo_coeff_orth_old = np.loadtxt(chk_lines[b1+1:a1], dtype=np.complex128)
            else:
                rt_scf._scf.mo_coeff = np.loadtxt(chk_lines[2:a1], dtype=np.complex128)
        else:
            if rt_scf.prop == 'magnus_interpol':
                b0 = []
                for idx, line in enumerate(chk_lines):
                    if 'Beta' in line:
                        b0.append(idx)
                    if 'Fock Orth N12 dt' in line:
                        b1 = idx

                mo_alpha0 = np.loadtxt(chk_lines[3:b0[0]], dtype=np.complex128)
                mo_beta0 = np.loadtxt(chk_lines[b0[0]+1:b1], dtype=np.complex128)
                rt_scf._scf.mo_coeff = np.stack((mo_alpha0, mo_beta0))

                fock_orth_n12dt_alpha = np.loadtxt(chk_lines[b1+2:b0[1]], dtype=np.complex128)
                fock_orth_n12dt_beta = np.loadtxt(chk_lines[b0[1]+1:a1], dtype=np.complex128)
                rt_scf._fock_orth_n12dt = np.stack((fock_orth_n12dt_alpha, fock_orth_n12dt_beta))
            elif rt_scf.prop == 'magnus_step':
                b0 = []
                for idx, line in enumerate(chk_lines):
                    if 'Beta' in line:
                        b0.append(idx)
                    if 'MO Coeff Orth Old' in line:
                        b1 = idx

                mo_alpha0 = np.loadtxt(chk_lines[3:b0[0]], dtype=np.complex128)
                mo_beta0 = np.loadtxt(chk_lines[b0[0]+1:b1], dtype=np.complex128)
                rt_scf._scf.mo_coeff = np.stack((mo_alpha0, mo_beta0))

                mo_coeff_orth_old_alpha = np.loadtxt(chk_lines[b1+2:b0[1]], dtype=np.complex128)
                mo_coeff_orth_old_beta = np.loadtxt(chk_lines[b0[1]+1:a1], dtype=np.complex128)
                rt_scf.mo_coeff_orth_old = np.stack((mo_coeff_orth_old_alpha, mo_coeff_orth_old_beta))
            else:
                for idx, line in enumerate(chk_lines):
                    if 'Beta' in line:
                        b0 = idx
                        break

                mo_alpha0 = np.loadtxt(chk_lines[3:b0], dtype=np.complex128)
                mo_beta0 = np.loadtxt(chk_lines[b0+1:], dtype=np.complex128)
                rt_scf._scf.mo_coeff = np.stack((mo_alpha0, mo_beta0))
        if hasattr(rt_scf, "nuc") and b1 is not None:
            rt_scf.nuc.pos = np.loadtxt(chk_lines[a1+1:a2], dtype=np.float64)
            rt_scf.nuc.vel = np.loadtxt(chk_lines[a2+1:a3], dtype=np.float64)
            rt_scf.nuc.force = np.loadtxt(chk_lines[a3+1:], dtype=np.float64)

def update_chkfile(rt_scf):
    with open(rt_scf.chkfile, 'w') as f:
        f.write(f'Current Time (AU): {rt_scf.current_time} \nMO Coeffs: \n')
        if rt_scf.nmat == 1:
            np.savetxt(f, rt_scf._scf.mo_coeff)

            # If using magnus_interpol, save _fock_orth_n12_dt
            if rt_scf.prop == 'magnus_interpol':
                f.write(f'\nFock Orth N12 dt: \n')
                np.savetxt(f, rt_scf._fock_orth_n12dt)
            # If using magnus_step, save mo_coeff_orth_old
            if rt_scf.prop == 'magnus_step':
                f.write(f'\nMO Coeff Orth Old: \n')
                np.savetxt(f, rt_scf.mo_coeff_orth_old)
        else:
            f.write('Alpha \n')
            np.savetxt(f, rt_scf._scf.mo_coeff[0])
            f.write('Beta \n')
            np.savetxt(f, rt_scf._scf.mo_coeff[1])

            # If using magnus_interpol, save _fock_orth_n12_dt
            if rt_scf.prop == 'magnus_interpol':
                f.write(f'\nFock Orth N12 dt: \n')
                f.write('Alpha \n')
                np.savetxt(f, rt_scf._fock_orth_n12dt[0])
                f.write('Beta \n')
                np.savetxt(f, rt_scf._fock_orth_n12dt[1])
            # If using magnus_step, save mo_coeff_orth_old
            if rt_scf.prop == 'magnus_step':
                f.write(f'\nMO Coeff Orth Old: \n')
                f.write('Alpha \n')
                np.savetxt(f, rt_scf.mo_coeff_orth_old[0])
                f.write('Beta \n')
                np.savetxt(f, rt_scf.mo_coeff_orth_old[1])
        if rt_scf.istype('RT_Ehrenfest'):
            f.write('Position (AU): \n')
            np.savetxt(f, rt_scf.nuc.pos)
            f.write('Velocity (AU): \n')
            np.savetxt(f, rt_scf.nuc.vel)
            f.write('Force (AU): \n')
            np.savetxt(f, rt_scf.nuc.force)


def print_info(rt_scf, mo_coeff_print):
    rt_scf._log.note(f'{"=" * 25} \nBeginning Propagation For: \n')
    scf_type = type(rt_scf._scf).__name__
    rt_scf._log.note(f'\t Object Type: {scf_type}')
    rt_scf._log.note(f'\t Basis Set: {rt_scf._scf.mol.basis}')
    rt_scf._log.note(f'\t Mol Length: {len(rt_scf._scf.mol._atom)}\n')
    if hasattr(rt_scf._scf, 'xc'):
        rt_scf._log.note(f'\t Exchange-Correlation Functional: {rt_scf._scf.xc}')
    if hasattr(rt_scf._scf, 'nlc') and rt_scf._scf.nlc != '':
        rt_scf._log.note(f'\t Non-local Dispersion Correction: {rt_scf._scf.nlc}')
    if hasattr(rt_scf._scf, 'disp') and rt_scf._scf.disp != None:
        rt_scf._log.note(f'\t Dispersion Correction: {rt_scf._scf.disp}')

    rt_scf._log.note('\nPropagation Settings: \n')
    if rt_scf.istype('RT_Ehrenfest'):
        rt_scf._log.note(f'\t Real-Time SCF w/ Ehrenfest Dynamics')
    else:
        rt_scf._log.note(f'\t Real-Time SCF')
    rt_scf._log.note(f'\t Integrator: {rt_scf.prop}')
    rt_scf._log.note(f'\t Max time (AU): {rt_scf.max_time}')
    rt_scf._log.note(f'\t Time step (AU): {rt_scf.timestep}')
    if rt_scf.istype('RT_Ehrenfest'):
        rt_scf._log.note(f'\t Nuclear Position Update Frequency: {rt_scf.Ne_step}')
        rt_scf._log.note(f'\t Nuclear Force Update Frequency: {rt_scf.N_step}')
    rt_scf._log.note(f'\t Observables: \n')
    for obs in rt_scf.observables.keys():
        if rt_scf.observables[obs]:
            rt_scf._log.note(f' \t \t {obs}')
    
    if rt_scf.observables['mo_occ'] or rt_scf.observables['mo_occ_separate']:
        if mo_coeff_print is None:
            if hasattr(rt_scf, 'mo_coeff_print'):
                rt_scf._log.note('\n\tPrinting molecular orbital occupations in the basis of self.mo_coeff_print\n')
            else:
                rt_scf._log.note('\n*** mo_coeff_print unspecified. Molecular orbital occupations will be printed in the basis of initial mo_coeff. ***\n')

                rt_scf.mo_coeff_print = rt_scf._scf.mo_coeff
        else:
            rt_scf.mo_coeff_print = mo_coeff_print

    if rt_scf._potential:
        rt_scf._log.note('\nApplied Potentials: \n')
        for vapp in rt_scf._potential:
            rt_scf._log.note(f'\t \t {type(vapp).__name__}')
    rt_scf._log.note('\n')

def _sym_orth(rt_ehrenfest):
    # Symmetrical orthogonalization is used for Ehrenfest dynamics
    rt_ehrenfest.evals, rt_ehrenfest.evecs = np.linalg.eigh(rt_ehrenfest.ovlp)
    return np.linalg.multi_dot([rt_ehrenfest.evecs, np.diag(np.power(rt_ehrenfest.evals, -0.5)), rt_ehrenfest.evecs.T])
