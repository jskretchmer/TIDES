import numpy as np
from tides import rt_output
from tides.basis_utils import _mask_fragment_basis
from tides.hirshfeld import hirshfeld_partition, get_weights
from tides.rt_utils import _update_mo_coeff_print
from pyscf import lib
from pyscf.lib import logger
from pyscf.tools import cubegen
from datetime import datetime
import os

'''
Real-time Observable Functions
'''

def _init_observables(rt_scf):
    rt_scf.observables = {
        'energy'               : False,
        'dipole'               : False,
        'quadrupole'           : False,
        'charge'               : False,
        'mulliken_charge'      : False,
        'mulliken_atom_charge' : False,
        'hirsh_charge'         : False,
        'hirsh_atom_charge'    : False,
        'mag'                  : False,
        'hirsh_mag'            : False,
        'hirsh_atom_mag'       : False,
        'mo_occ'               : False,
        'mo_occ_separate'      : False,
        'nuclei'               : False,
        'cube_density'         : False,
        'spin_square'          : False,
        'mo_coeff'             : False,
        'den_ao'               : False,
        'fock_ao'              : False,
        }

    rt_scf._observables_functions = {
        'energy'               : [get_energy, rt_output._print_energy],
        'dipole'               : [get_dipole, rt_output._print_dipole],
        'quadrupole'           : [get_quadrupole, rt_output._print_quadrupole],
        'charge'               : [get_charge, rt_output._print_charge],
        'mulliken_charge'      : [get_mulliken_charge, rt_output._print_mulliken_charge],
        'mulliken_atom_charge' : [get_mulliken_charge, rt_output._print_mulliken_charge],
        'hirsh_charge'         : [get_hirshfeld_charge, rt_output._print_hirshfeld_charge],
        'hirsh_atom_charge'    : [get_hirshfeld_charge, rt_output._print_hirshfeld_charge],
        'mag'                  : [get_mag, rt_output._print_mag],
        'hirsh_mag'            : [get_hirshfeld_mag, rt_output._print_hirshfeld_mag],
        'hirsh_atom_mag'       : [get_hirshfeld_mag, rt_output._print_hirshfeld_mag],
        'mo_occ'               : [get_mo_occ, rt_output._print_mo_occ],
        'mo_occ_separate'      : [get_mo_occ_separate, rt_output._print_mo_occ_separate],
        'nuclei'               : [get_nuclei, rt_output._print_nuclei],
        'cube_density'         : [get_cube_density, lambda *args: None],
        'spin_square'          : [get_spin_square, rt_output._print_spin_square],
        'mo_coeff'             : [lambda *args: None, rt_output._print_mo_coeff],
        'den_ao'               : [lambda *args: None, rt_output._print_den_ao],
        'fock_ao'              : [lambda *args: None, rt_output._print_fock_ao],
        }



def _check_observables(rt_scf):
    if rt_scf.observables['mag'] | rt_scf.observables['hirsh_atom_mag']:
        assert rt_scf._scf.istype('GHF') | rt_scf._scf.istype('GKS')

    # Get atomic weights if using Hirshfeld Scheme
    if (rt_scf.observables['hirsh_atom_mag'] | rt_scf.observables['hirsh_mag'] |
    rt_scf.observables['hirsh_atom_charge'] | rt_scf.observables['hirsh_charge']):
        rt_scf.hirshfeld = True
        rt_scf.grids, rt_scf.atom_weights = get_weights(rt_scf._scf.mol)
    else:
        rt_scf.hirshfeld = False

    ### For whatever reason, the dip_moment call for GHF and GKS has arg name 'unit_symbol' instead of 'unit'
    if rt_scf._scf.istype('GHF') | rt_scf._scf.istype('GKS'):
        rt_scf._observables_functions['dipole'][0] = _temp_get_dipole

    # mo_occ_separate should be used for spin unrestricted when use wants to know specifically alpha vs. beta occupations
    if rt_scf.observables['mo_occ_separate']:
        assert rt_scf._scf.istype('UHF') | rt_scf._scf.istype('UKS')

    # If we are printing nuclei, we must be a RT_Ehrenfest object
    if rt_scf.observables['nuclei']:
        assert rt_scf.istype('RT_Ehrenfest')

    # If we are printing nuclei, we need to generate an xyz file
    if rt_scf.observables['nuclei']:
        xyz_file = 'trajectory'
        
        # Check to make sure this file doesn't exist
        name_available = False
        while not name_available:
            if os.path.exists(f'{xyz_file}.xyz'):
                # Just append the current time to the filename if trajectory.xyz already exists
                xyz_file = xyz_file + str(datetime.now()).replace(' ','_').replace('-','_').replace(':','_').split('.')[0]
            else:
                name_available = True
                break

        rt_scf._xyz_fh = open(f'{xyz_file}.xyz', 'w') 
        rt_scf._xyz_log = logger.Logger(rt_scf._xyz_fh, verbose=rt_scf.verbose)
        rt_scf._log.note(f'Nuclei xyz output file: {xyz_file}.xyz')

        # rt_scf.verbose > 2 will print coordinates
        # rt_scf.verbose > 3 will print coordinates and velocities
        # rt_scf.verbose > 4 will print coordinates and velocities and forces
        # Checking here to assign to correct print function for the entire calculation, 
        # so we don't have to check every time step.
        if rt_scf.verbose > 4:
            rt_scf._update_xyz = rt_output._nuclei_coords_vels_forces
        elif rt_scf.verbose > 3:
            rt_scf._update_xyz = rt_output._nuclei_coords_vels
        elif rt_scf.verbose > 2:
            rt_scf._update_xyz = rt_output._nuclei_coords

    for key, print_value in rt_scf.observables.items():
        if not print_value:
            del rt_scf._observables_functions[key]

    # Now rt_scf._observables_functions only has observables to be calculated
    # One final check here that if any observable is being calculated, rt_scf.verbose > 2
    # Likely no one will ever run with verbose < 3 (no observables will be printed)
    if rt_scf._observables_functions:
        assert rt_scf.verbose > 2

def get_observables(rt_scf):
    if rt_scf.istype('RT_Ehrenfest'):
        if 'mo_occ' in rt_scf.observables or 'mo_occ_separate' in rt_scf.observables:
            _update_mo_coeff_print(rt_scf)
        if rt_scf.hirshfeld:
            rt_scf.grids, rt_scf.atom_weights = get_weights(rt_scf._scf.mol)

    for key, function in rt_scf._observables_functions.items():
          function[0](rt_scf, rt_scf.den_ao)

    rt_output.update_output(rt_scf)

def get_energy(rt_scf, den_ao):
    rt_scf._energy = []
    rt_scf._energy.append(rt_scf._scf.energy_tot(dm=den_ao))
    if rt_scf.istype('RT_Ehrenfest'):
        ke = rt_scf.nuc.get_ke()
        rt_scf._energy[0] += np.sum(ke)
        rt_scf._kinetic_energy = ke
    for frag in rt_scf.fragments:
        rt_scf._energy.append(frag.energy_tot(dm=den_ao[frag.mask]))
        if rt_scf.istype('RT_Ehrenfest'):
            rt_scf._energy[-1] += np.sum(ke[frag.match_indices])


def get_charge(rt_scf, den_ao):
    # charge = tr(PaoS)
    rt_scf._charge = []
    if rt_scf.nmat == 2:
        rt_scf._charge.append(np.trace(np.sum(np.matmul(den_ao,rt_scf.ovlp), axis=0)))
        for frag in rt_scf.fragments:
            rt_scf._charge.append(np.trace(np.sum(np.matmul(den_ao,rt_scf.ovlp)[frag.mask], axis=0)))
    else:
        rt_scf._charge.append(np.trace(np.matmul(den_ao,rt_scf.ovlp)))
        for frag in rt_scf.fragments:
            rt_scf._charge.append(np.trace(np.matmul(den_ao,rt_scf.ovlp)[frag.mask]))

def get_mulliken_charge(rt_scf, den_ao):
    rt_scf._atom_charges = []
    if rt_scf.nmat == 2:
        for idx, label in enumerate(rt_scf._scf.mol._atom):
            atom_mask = _mask_fragment_basis(rt_scf._scf, [idx])
            rt_scf._atom_charges.append(np.trace(np.sum(np.matmul(den_ao,rt_scf.ovlp)[atom_mask], axis=0)))
    else:
        for idx, label in enumerate(rt_scf._scf.mol._atom):
            atom_mask = _mask_fragment_basis(rt_scf._scf, [idx])
            rt_scf._atom_charges.append(np.trace(np.matmul(den_ao,rt_scf.ovlp)[atom_mask]))

def get_hirshfeld_charge(rt_scf, den_ao):
    if rt_scf.nmat == 2:
        rho_a, rho_b = hirshfeld_partition(rt_scf._scf, den_ao, rt_scf.grids, rt_scf.atom_weights)
        rho = rho_a + rho_b
    elif rt_scf._scf.istype('GHF') | rt_scf._scf.istype('GKS'):
        rho_aa, rho_ab, rho_ba, rho_bb = hirshfeld_partition(rt_scf._scf, den_ao, rt_scf.grids, rt_scf.atom_weights)
        rho = rho_aa + rho_bb
    else:
        rho = hirshfeld_partition(rt_scf._scf, den_ao, rt_scf.grids, rt_scf.atom_weights)
    rt_scf._hirshfeld_charges = rho.sum(axis=1)

def get_dipole(rt_scf, den_ao):
    rt_scf._dipole = rt_scf._scf.dip_moment(mol=rt_scf._scf.mol, dm=rt_scf.den_ao, unit='A.U.', verbose=1)

def _temp_get_dipole(rt_scf, den_ao):
    # Temporary fix for argument name discrepancy in GHF.dip_moment ('unit_symbol' instead of 'unit')
    rt_scf._dipole = rt_scf._scf.dip_moment(mol=rt_scf._scf.mol, dm=rt_scf.den_ao, unit_symbol='A.U.', verbose=1)

def get_quadrupole(rt_scf, den_ao):
    rt_scf._quadrupole = rt_scf._scf.quad_moment(mol=rt_scf._scf.mol, dm=rt_scf.den_ao,unit='A.U.', verbose=1)

def get_mag(rt_scf, den_ao):
    Nsp = int(np.shape(rt_scf.ovlp)[0] / 2)

    magx = np.sum((den_ao[:Nsp, Nsp:] + den_ao[Nsp:, :Nsp]) * rt_scf.ovlp[:Nsp,:Nsp])
    magy = 1j * np.sum((den_ao[:Nsp, Nsp:] - den_ao[Nsp:, :Nsp]) * rt_scf.ovlp[:Nsp,:Nsp])
    magz = np.sum((den_ao[:Nsp, :Nsp] - den_ao[Nsp:, Nsp:]) * rt_scf.ovlp[:Nsp,:Nsp])
    rt_scf._mag = [magx, magy, magz]

def get_hirshfeld_mag(rt_scf, den_ao):
    rho_aa, rho_ab, rho_ba, rho_bb = hirshfeld_partition(rt_scf._scf, den_ao, rt_scf.grids, rt_scf.atom_weights)
    mx = (rho_ab + rho_ba)
    my = 1j * (rho_ab - rho_ba)
    mz = (rho_aa - rho_bb)

    rt_scf._hirshfeld_mx_atoms = mx.sum(axis=1)
    rt_scf._hirshfeld_my_atoms = my.sum(axis=1)
    rt_scf._hirshfeld_mz_atoms = mz.sum(axis=1)

def get_mo_occ(rt_scf, den_ao):
    # P_mo = C+SP_aoSC
    SP_aoS = np.matmul(rt_scf.ovlp,np.matmul(den_ao,rt_scf.ovlp))
    if rt_scf.nmat == 2:
        mo_coeff_print_transpose = np.stack((rt_scf.mo_coeff_print[0].T.conj(), rt_scf.mo_coeff_print[1].T.conj()))
        den_mo = np.matmul(mo_coeff_print_transpose,np.matmul(SP_aoS,rt_scf.mo_coeff_print))
        den_mo = np.real(np.sum(den_mo,axis=0))
    else:
        den_mo = np.matmul(rt_scf.mo_coeff_print.T.conj(), np.matmul(SP_aoS,rt_scf.mo_coeff_print))
        den_mo = np.real(den_mo)

    rt_scf._mo_occ = np.diagonal(den_mo)

def get_mo_occ_separate(rt_scf, den_ao):
    # P_mo = C+SP_aoSC
    SP_aoS = np.matmul(rt_scf.ovlp,np.matmul(den_ao,rt_scf.ovlp))
    mo_coeff_print_transpose = np.stack((rt_scf.mo_coeff_print[0].T.conj(), rt_scf.mo_coeff_print[1].T.conj()))
    den_mo = np.matmul(mo_coeff_print_transpose,np.matmul(SP_aoS,rt_scf.mo_coeff_print))
    den_mo = np.real(den_mo)

    rt_scf._mo_occ_separate = [np.diagonal(den_mo[0]), np.diagonal(den_mo[1])]

def get_nuclei(rt_scf, den_ao):
    rt_scf._nuclei = [rt_scf.nuc.labels, rt_scf.nuc.pos*lib.param.BOHR, rt_scf.nuc.vel*lib.param.BOHR, rt_scf.nuc.force]

def get_cube_density(rt_scf, den_ao):
    '''
    Will create Gaussian cube file for molecule electron density
    for every propagation time given in rt_scf.cube_density_indices.
    '''
    if np.rint(rt_scf.current_time/rt_scf.timestep) in np.rint(np.array(rt_scf.cube_density_indices)/rt_scf.timestep):
        if hasattr(rt_scf, 'cube_filename'):
            cube_name = f'{rt_scf.cube_filename}{rt_scf.current_time}.cube'
        else:
            cube_name = f'{rt_scf.current_time}.cube'
        cubegen.density(rt_scf._scf.mol, cube_name, den_ao)

def get_spin_square(rt_scf, den_ao):
    if rt_scf._scf.istype('UHF'):
        mo_coeff = (rt_scf._scf.mo_coeff[0][:,rt_scf.occ[0]>0],
                    rt_scf._scf.mo_coeff[1][:,rt_scf.occ[1]>0])
    else:
        mo_coeff = rt_scf._scf.mo_coeff[:,rt_scf.occ>0]

    rt_scf._s2, rt_scf._2s_p1 = rt_scf._scf.spin_square(mo_coeff, s=rt_scf.ovlp)
