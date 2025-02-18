import numpy as np
from tides import rt_output
from tides.basis_utils import read_mol, mask_fragment_basis
from tides.hirshfeld import hirshfeld_partition, get_weights
from tides.rt_utils import update_mo_coeff_print
from pyscf import lib
from pyscf.tools import cubegen

'''
Real-time Observable Functions
'''

def _init_observables(rt_mf):
    rt_mf.observables = {
        'energy'               : False,
        'dipole'               : False,
        'quadrupole'           : False,
        'charge'               : False,
        'atom_charge'          : False,
        'mulliken_charge'      : False,
        'mulliken_atom_charge' : False,
        'hirsh_atom_charge'    : False,
        'mag'                  : False,
        'hirsh_atom_mag'       : False,
        'mo_occ'               : False,
        'atom_charge'          : False,
        'nuclei'               : False,
        'cube_density'         : False,
        'mo_coeff'             : False,
        'den_ao'               : False,
        'fock_ao'              : False,
        }

    rt_mf._observables_functions = {
        'energy'               : [get_energy, rt_output._print_energy],
        'dipole'               : [get_dipole, rt_output._print_dipole],
        'quadrupole'           : [get_quadrupole, rt_output._print_quadrupole],
        'charge'               : [get_charge, rt_output._print_charge],
        'atom_charge'          : [get_atom_charge, rt_output._print_atom_charge],
        'mulliken_charge'      : [get_charge, rt_output._print_charge],
        'mulliken_atom_charge' : [get_atom_charge, rt_output._print_atom_charge],
        'hirsh_atom_charge'    : [get_hirshfeld_charge, rt_output._print_hirshfeld_charge],
        'mag'                  : [get_mag, rt_output._print_mag],
        'hirsh_atom_mag'       : [get_hirshfeld_mag, rt_output._print_hirshfeld_mag],
        'mo_occ'               : [get_mo_occ, rt_output._print_mo_occ],
        'atom_charge'          : [get_atom_charge, rt_output._print_atom_charge],
        'nuclei'               : [get_nuclei, rt_output._print_nuclei],
        'cube_density'         : [get_cube_density, lambda *args: None],
        'mo_coeff'             : [lambda *args: None, rt_output._print_mo_coeff],
        'den_ao'               : [lambda *args: None, rt_output._print_den_ao],
        'fock_ao'              : [lambda *args: None, rt_output._print_fock_ao],
        }



def _check_observables(rt_mf):
    if rt_mf.observables['mag'] | rt_mf.observables['hirsh_atom_mag']:
        assert rt_mf._scf.istype('GHF') | rt_mf._scf.istype('GKS')

    # Get atomic weights if using Hirshfeld Scheme
    if rt_mf.observables['hirsh_atom_mag'] | rt_mf.observables['hirsh_atom_charge']:
        rt_mf.hirshfeld = True
        rt_mf.grids, rt_mf.atom_weights = get_weights(rt_mf._scf.mol)
    else:
        rt_mf.hirshfeld = False

    ### For whatever reason, the dip_moment call for GHF and GKS has arg name 'unit_symbol' instead of 'unit'
    if rt_mf._scf.istype('GHF') | rt_mf._scf.istype('GKS'):
        rt_mf._observables_functions['dipole'][0] = temp_get_dipole

    for key, print_value in rt_mf.observables.items():
        if not print_value:
            del rt_mf._observables_functions[key]



def get_observables(rt_mf):
    if rt_mf.istype('RT_Ehrenfest'):
        if 'mo_occ' in rt_mf.observables:
            update_mo_coeff_print(rt_mf)
        if rt_mf.hirshfeld:
            rt_mf.grids, rt_mf.atom_weights = get_weights(rt_mf._scf.mol)

    for key, function in rt_mf._observables_functions.items():
          function[0](rt_mf, rt_mf.den_ao)

    rt_output.update_output(rt_mf)

def get_energy(rt_mf, den_ao):
    rt_mf._energy = []
    rt_mf._energy.append(rt_mf._scf.energy_tot(dm=den_ao))
    if rt_mf.istype('RT_Ehrenfest'):
        ke = rt_mf.nuc.get_ke()
        rt_mf._energy[0] += np.sum(ke)
        rt_mf._kinetic_energy = ke
    for frag in rt_mf.fragments:
        rt_mf._energy.append(frag.energy_tot(dm=den_ao[frag.mask]))
        if rt_mf.istype('RT_Ehrenfest'):
            rt_mf._energy[-1] += np.sum(ke[frag.match_indices])


def get_charge(rt_mf, den_ao):
    # charge = tr(PaoS)
    rt_mf._charge = []
    if rt_mf.nmat == 2:
        rt_mf._charge.append(np.trace(np.sum(np.matmul(den_ao,rt_mf.ovlp), axis=0)))
        for frag in rt_mf.fragments:
            rt_mf._charge.append(np.trace(np.sum(np.matmul(den_ao,rt_mf.ovlp)[frag.mask], axis=0)))
    else:
        rt_mf._charge.append(np.trace(np.matmul(den_ao,rt_mf.ovlp)))
        for frag in rt_mf.fragments:
            rt_mf._charge.append(np.trace(np.matmul(den_ao,rt_mf.ovlp)[frag.mask]))

def get_hirshfeld_charge(rt_mf, den_ao):
    if rt_mf.nmat == 2:
        rho_a, rho_b = hirshfeld_partition(rt_mf._scf, den_ao, rt_mf.grids, rt_mf.atom_weights)
        rho = rho_a + rho_b
    elif rt_mf._scf.istype('GHF') | rt_mf._scf.istype('GKS'):
        rho_aa, rho_ab, rho_ba, rho_bb = hirshfeld_partition(rt_mf._scf, den_ao, rt_mf.grids, rt_mf.atom_weights)
        rho = rho_aa + rho_bb
    else:
        rho = hirshfeld_partition(rt_mf._scf, den_ao, rt_mf.grids, rt_mf.atom_weights)
    rt_mf._hirshfeld_charges = rho.sum(axis=1)

def get_dipole(rt_mf, den_ao):
    rt_mf._dipole = []
    rt_mf._dipole.append(rt_mf._scf.dip_moment(mol=rt_mf._scf.mol, dm=rt_mf.den_ao, unit='A.U.', verbose=1))
    for frag in rt_mf.fragments:
        rt_mf._dipole.append(frag.dip_moment(mol=frag.mol, dm=den_ao[frag.mask], unit='A.U.', verbose=1))

def temp_get_dipole(rt_mf, den_ao):
    # Temporary fix for argument name discrepancy in GHF.dip_moment ('unit_symbol' instead of 'unit')
    rt_mf._dipole = []
    rt_mf._dipole.append(rt_mf._scf.dip_moment(mol=rt_mf._scf.mol, dm=rt_mf.den_ao, unit_symbol='A.U.', verbose=1))
    for frag in rt_mf.fragments:
        rt_mf._dipole.append(frag.dip_moment(mol=frag.mol, dm=den_ao[frag.mask], unit_symbol='A.U.', verbose=1))

def get_quadrupole(rt_mf, den_ao):
    rt_mf._quadrupole = []
    rt_mf._quadrupole.append(rt_mf._scf.quad_moment(mol=rt_mf._scf.mol, dm=rt_mf.den_ao,unit='A.U.', verbose=1))
    for frag in rt_mf.fragments:
        rt_mf._quadrupole.append(frag.quad_moment(mol=frag.mol, dm=den_ao[frag.mask], unit='A.U.', verbose=1))

def get_mo_occ(rt_mf, den_ao):
    # P_mo = C+SP_aoSC
    SP_aoS = np.matmul(rt_mf.ovlp,np.matmul(den_ao,rt_mf.ovlp))
    if rt_mf.nmat == 2:
        mo_coeff_print_transpose = np.stack((rt_mf.mo_coeff_print[0].T, rt_mf.mo_coeff_print[1].T))
        den_mo = np.matmul(mo_coeff_print_transpose,np.matmul(SP_aoS,rt_mf.mo_coeff_print))
        den_mo = np.real(np.sum(den_mo,axis=0))
    else:
        den_mo = np.matmul(rt_mf.mo_coeff_print.T, np.matmul(SP_aoS,rt_mf.mo_coeff_print))
        den_mo = np.real(den_mo)

    rt_mf._mo_occ = np.diagonal(den_mo)

def get_mag(rt_mf, den_ao):
    rt_mf._mag = []
    Nsp = int(np.shape(rt_mf.ovlp)[0] / 2)

    magx = np.sum((den_ao[:Nsp, Nsp:] + den_ao[Nsp:, :Nsp]) * rt_mf.ovlp[:Nsp,:Nsp])
    magy = 1j * np.sum((den_ao[:Nsp, Nsp:] - den_ao[Nsp:, :Nsp]) * rt_mf.ovlp[:Nsp,:Nsp])
    magz = np.sum((den_ao[:Nsp, :Nsp] - den_ao[Nsp:, Nsp:]) * rt_mf.ovlp[:Nsp,:Nsp])
    rt_mf._mag.append([magx, magy, magz])

def get_hirshfeld_mag(rt_mf, den_ao):
    rho_aa, rho_ab, rho_ba, rho_bb = hirshfeld_partition(rt_mf._scf, den_ao, rt_mf.grids, rt_mf.atom_weights)
    mx = (rho_ab + rho_ba)
    my = 1j * (rho_ab - rho_ba)
    mz = (rho_aa - rho_bb)

    rt_mf._hirshfeld_mx_atoms = mx.sum(axis=1)
    rt_mf._hirshfeld_my_atoms = my.sum(axis=1)
    rt_mf._hirshfeld_mz_atoms = mz.sum(axis=1)

def get_atom_charge(rt_mf, den_ao):
    rt_mf._atom_charges = []
    if rt_mf.nmat == 2:
        for idx, label in enumerate(rt_mf._scf.mol._atom):
            atom_mask = mask_fragment_basis(rt_mf._scf, [idx])
            rt_mf._atom_charges.append(np.trace(np.sum(np.matmul(den_ao,rt_mf.ovlp)[atom_mask], axis=0)))
    else:
        for idx, label in enumerate(rt_mf._scf.mol._atom):
            atom_mask = mask_fragment_basis(rt_mf._scf, [idx])
            rt_mf._atom_charges.append(np.trace(np.matmul(den_ao,rt_mf.ovlp)[atom_mask]))

def get_nuclei(rt_mf, den_ao):
    rt_mf._nuclei = [rt_mf.nuc.labels, rt_mf.nuc.pos*lib.param.BOHR, rt_mf.nuc.vel*lib.param.BOHR, rt_mf.nuc.force]

def get_cube_density(rt_mf, den_ao):
    '''
    Will create Gaussian cube file for molecule electron density
    for every propagation time given in rt_mf.cube_density_indices.
    '''
    if np.rint(rt_mf.current_time/rt_mf.timestep) in np.rint(np.array(rt_mf.cube_density_indices)/rt_mf.timestep):
        if hasattr(rt_mf, 'cube_filename'):
            cube_name = f'{rt_mf.cube_filename}{rt_mf.current_time}.cube'
        else:
            cube_name = f'{rt_mf.current_time}.cube'
        cubegen.density(rt_mf._scf.mol, cube_name, den_ao)
