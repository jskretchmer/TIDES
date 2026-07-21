import numpy as np
from tides.observables import rt_output
from tides.utils.basis_utils import _mask_fragment_basis
from tides.observables.hirshfeld import hirshfeld_partition, get_weights
from tides.utils.rt_utils import _update_mo_coeff_print
from pyscf import lib
from pyscf.lib import logger
from pyscf.tools import cubegen
import os
from datetime import datetime


'''
Real-time Observable Functions
'''

def _init_observables(rt_obj):
    rt_obj.observables = {
        'energy'               : False,
        'dipole'               : False,
        'quadrupole'           : False,
        'charge'               : False,
        'mulliken_charge'      : False,
        'mulliken_atom_charge' : False,
        'hirsh_charge'         : False,
        'hirsh_atom_charge'    : False,
        'hirshi_atom_charge'   : False, #HORTON Hirshfeld-I
        'plane_partition_charge': False, # Plane partitioning charge
        'plane_partition_charge_spatial': False, # Plane partitioning charge using spatial integration
        'mag'                  : False,
        'hirsh_mag'            : False,
        'hirsh_atom_mag'       : False,
        'spin_square'          : False,
        'mo_occ'               : False,
        'mo_occ_separate'      : False,
        'nuclei'               : False,
        'cube_density'         : False,
        'mo_coeff'             : False,
        'den_ao'               : False,
        'fock_ao'              : False,
        }

    rt_obj._observables_functions = {
        'energy'               : [get_energy, rt_output._print_energy],
        'dipole'               : [get_dipole, rt_output._print_dipole],
        'quadrupole'           : [get_quadrupole, rt_output._print_quadrupole],
        'charge'               : [get_charge, rt_output._print_charge],
        'mulliken_charge'      : [get_mulliken_charge, rt_output._print_mulliken_charge],
        'mulliken_atom_charge' : [get_mulliken_charge, rt_output._print_mulliken_charge],
        'hirsh_charge'         : [get_hirshfeld_charge, rt_output._print_hirshfeld_charge],
        'hirsh_atom_charge'    : [get_hirshfeld_charge, rt_output._print_hirshfeld_charge],
        'hirshi_atom_charge'   : [get_hirshfeld_i_charge, rt_output._print_hirshfeld_charge],
        'plane_partition_charge': [get_plane_partition_charge, rt_output._print_plane_partition_charge],
        'plane_partition_charge_spatial': [get_plane_partition_charge_spatial, rt_output._print_plane_partition_charge_spatial],
        'mag'                  : [get_mag, rt_output._print_mag],
        'hirsh_mag'            : [get_hirshfeld_mag, rt_output._print_hirshfeld_mag],
        'hirsh_atom_mag'       : [get_hirshfeld_mag, rt_output._print_hirshfeld_mag],
        'mo_occ'               : [get_mo_occ, rt_output._print_mo_occ],
        'mo_occ_separate'      : [get_mo_occ_separate, rt_output._print_mo_occ_separate],
        'nuclei'               : [get_nuclei, rt_output._print_nuclei],
        'cube_density'         : [get_cube_density, lambda *args: None],
        'mo_coeff'             : [lambda *args: None, rt_output._print_mo_coeff],
        'den_ao'               : [lambda *args: None, rt_output._print_den_ao],
        'fock_ao'              : [lambda *args: None, rt_output._print_fock_ao],
        'spin_square'          : [get_spin_square, rt_output._print_spin_square],
        }



def _check_observables(rt_obj):
    if rt_obj.observables['mag'] | rt_obj.observables['hirsh_atom_mag']:
        assert rt_obj._scf.istype('GHF') | rt_obj._scf.istype('GKS')

    # Get atomic weights if using Hirshfeld Scheme
    if (rt_obj.observables['hirsh_atom_mag'] | rt_obj.observables['hirsh_mag'] |
    rt_obj.observables['hirsh_atom_charge'] | rt_obj.observables['hirsh_charge']):
        rt_obj.hirshfeld = True
        rt_obj.grids, rt_obj.atom_weights = get_weights(rt_obj._scf.mol)
    else:
        rt_obj.hirshfeld = False

    ### For whatever reason, the dip_moment call for GHF and GKS has arg name 'unit_symbol' instead of 'unit'
    if rt_obj._scf.__class__.__name__ != 'CASCI' and rt_obj._scf.__class__.__name__ != 'CASSCF':
        if rt_obj._scf.istype('GHF') | rt_obj._scf.istype('GKS'):
            rt_obj._observables_functions['dipole'][0] = _temp_get_dipole

    # mo_occ_separate should be used for spin unrestricted when use wants to know specifically alpha vs. beta occupations
    if rt_obj.observables['mo_occ_separate']:
        assert rt_obj._scf.istype('UHF') | rt_obj._scf.istype('UKS')

    # mo_occ_separate should be used for spin unrestricted when use wants to know specifically alpha vs. beta occupations
    if rt_obj.observables['mo_occ_separate']:
        assert rt_obj._scf.istype('UHF') | rt_obj._scf.istype('UKS')

    # If we are printing nuclei, we must be a RT_Ehrenfest object
    if rt_obj.observables['nuclei']:
        assert rt_obj.istype('RT_Ehrenfest')

    if rt_obj.observables['nuclei']:
        xyz_file = 'trajectory'

        name_available = False
        while not name_available:
            if os.path.exists(f'{xyz_file}.xyz'):
                # Just append the current time to the filename if trajectory.xyz already exists
                xyz_file = xyz_file + str(datetime.now()).replace(' ','_').replace('-','_').replace(':','_').split('.')[0]
            else:
                name_available = True
                break

        rt_obj._xyz_fh = open(f'{xyz_file}.xyz', 'w') 
        rt_obj._xyz_log = logger.Logger(rt_obj._xyz_fh, verbose=rt_obj.verbose)
        rt_obj._log.note(f'Nuclei xyz output file: {xyz_file}.xyz')

        # rt_obj.verbose > 2 will print coordinates
        # rt_obj.verbose > 3 will print coordinates and velocities
        # rt_obj.verbose > 4 will print coordinates and velocities and forces
        # Checking here to assign to correct print function for the entire calculation,
        # so we don't have to check every time step.

        if rt_obj.verbose > 4:
            rt_obj._update_xyz = rt_output._nuclei_coords_vels_forces
        elif rt_obj.verbose > 3:
            rt_obj._update_xyz = rt_output._nuclei_coords_vels
        elif rt_obj.verbose > 2:
            rt_obj._update_xyz = rt_output._nuclei_coords

    for key, print_value in rt_obj.observables.items():
        if not print_value:
            del rt_obj._observables_functions[key]

    if rt_obj._observables_functions:
        assert rt_obj.verbose >2


def get_observables(rt_obj):
    if rt_obj.observables.get('plane_partition_charge', False):
        rt_obj.grids, rt_obj.atom_weights = get_weights(rt_obj._scf.mol)
    if rt_obj.istype('RT_Ehrenfest'):
        if 'mo_occ' in rt_obj.observables or 'mo_occ_separate' in rt_obj.observables:
            _update_mo_coeff_print(rt_obj)
        if rt_obj.hirshfeld:
            rt_obj.grids, rt_obj.atom_weights = get_weights(rt_obj._scf.mol)

    for key, function in rt_obj._observables_functions.items():
          function[0](rt_obj)
          function[1](rt_obj)

    rt_output.update_output(rt_obj)

def get_energy(rt_obj):
    rt_obj._energy = []
    if rt_obj._scf.__class__.__name__ != 'CASCI' and rt_obj._scf.__class__.__name__ != 'CASSCF':
        rt_obj._energy.append(rt_obj._scf.energy_tot(dm=rt_obj.den_ao))
    else:
        full_energy, cas_energy = rt_obj.get_full_e()
        rt_obj._energy.append(full_energy)
        rt_obj._cas_energy = cas_energy
    if rt_obj.istype('RT_Ehrenfest'):
        ke = rt_obj.nuc.get_ke()
        rt_obj._energy[0] += np.sum(ke)
        rt_obj._kinetic_energy = ke
    for frag in rt_obj.fragments:
        rt_obj._energy.append(frag.energy_tot(dm=rt_obj.den_ao[frag.mask]))
        if rt_obj.istype('RT_Ehrenfest'):
            rt_obj._energy[-1] += np.sum(ke[frag.match_indices])


def get_charge(rt_obj):
    # charge = tr(PaoS)
    rt_obj._charge = []
    if rt_obj.nmat == 2:
        rt_obj._charge.append(np.trace(np.sum(np.matmul(rt_obj.den_ao,rt_obj.ovlp), axis=0)))
        for frag in rt_obj.fragments:
            rt_obj._charge.append(np.trace(np.sum(np.matmul(rt_obj.den_ao,rt_obj.ovlp)[frag.mask], axis=0)))
    else:
        rt_obj._charge.append(np.trace(np.matmul(rt_obj.den_ao,rt_obj.ovlp)))
        for frag in rt_obj.fragments:
            rt_obj._charge.append(np.trace(np.matmul(rt_obj.den_ao,rt_obj.ovlp)[frag.mask]))

def get_mulliken_charge(rt_obj):
    rt_obj._atom_charges = []
    if rt_obj.nmat == 2:
        for idx, label in enumerate(rt_obj._scf.mol._atom):
            atom_mask = _mask_fragment_basis(rt_obj._scf, [idx])
            rt_obj._atom_charges.append(np.trace(np.sum(np.matmul(rt_obj.den_ao,rt_obj.ovlp)[atom_mask], axis=0)))
    else:
        for idx, label in enumerate(rt_obj._scf.mol._atom):
            atom_mask = _mask_fragment_basis(rt_obj._scf, [idx])
            rt_obj._atom_charges.append(np.trace(np.matmul(rt_obj.den_ao,rt_obj.ovlp)[atom_mask]))

def get_hirshfeld_charge(rt_obj):
    if rt_obj.nmat == 2:
        rho_a, rho_b = hirshfeld_partition(rt_obj._scf, rt_obj.den_ao, rt_obj.grids, rt_obj.atom_weights)
        rho = rho_a + rho_b
    elif rt_obj._scf.istype('GHF') | rt_obj._scf.istype('GKS'):
        rho_aa, rho_ab, rho_ba, rho_bb = hirshfeld_partition(rt_obj._scf, rt_obj.den_ao, rt_obj.grids, rt_obj.atom_weights)
        rho = rho_aa + rho_bb
    else:
        rho = hirshfeld_partition(rt_obj._scf, rt_obj.den_ao, rt_obj.grids, rt_obj.atom_weights)
    rt_obj._hirshfeld_charges = rho.sum(axis=1)

def get_hirshfeld_i_charge(rt_obj):
    """
    HORTON Hirshfeld-I charges; requires a ProAtomDB (atoms.h5).
    """
    import os
    from tides.observables.horton_part import compute_hirshfeld_i_from_pyscf, HortonUnavailableError, create_proatomdb_from_pyscf
    
    # Try to get the database path from the environment or rt_obj attribute
    padb_path = getattr(rt_obj, 'horton_proatomdb_path', None) or os.environ.get('HORTON_ATOMDB', None)
    
    try:
        if padb_path is None:
            # No database path provided, create one on-the-fly
            print("Creating proatomic database on-the-fly for Hirshfeld-I calculation...")
            
            # Create a file in the current directory
            current_dir = os.getcwd()
            padb_path = os.path.join(current_dir, "atoms_temp.h5")
            
            # Create the database using the molecule's basis set and methods
            proatomdb = create_proatomdb_from_pyscf(
                rt_obj._scf.mol,
                method='dft',
                xc=getattr(rt_obj._scf, 'xc', 'pbe'),
                filename=padb_path
            )
            
            # Pass the database object directly
            res = compute_hirshfeld_i_from_pyscf(rt_obj._scf, rt_obj.den_ao, proatomdb=proatomdb)
        else:
            # Use the provided database path
            res = compute_hirshfeld_i_from_pyscf(rt_obj._scf, rt_obj.den_ao, proatomdb=padb_path)
        
        # Store the results in hirshfeld_i_charges attribute
        charges = res["charges"]
        
        # Initialize if not already done
        if not hasattr(rt_obj, 'hirshfeld_i_charges'):
            rt_obj.hirshfeld_i_charges = []
            
        # Append the charges
        rt_obj.hirshfeld_i_charges.append(charges)
        
        # Also store in _hirshfeld_charges for compatibility with printing functions
        rt_obj._hirshfeld_charges = charges
        
    except HortonUnavailableError as e:
        # Fall back to MBIS if HORTON is not available
        print(f"Warning: {str(e)}. Falling back to MBIS charges.")
        from tides.observables.horton_part import compute_mbis_from_pyscf
        try:
            res = compute_mbis_from_pyscf(rt_obj._scf, rt_obj.den_ao)
            charges = res["charges"]
            if not hasattr(rt_obj, 'hirshfeld_i_charges'):
                rt_obj.hirshfeld_i_charges = []
            rt_obj.hirshfeld_i_charges.append(charges)
            rt_obj._hirshfeld_charges = charges
            print("Using MBIS charges instead of Hirshfeld-I.")
        except Exception as e2:
            raise RuntimeError(f"Both Hirshfeld-I and MBIS failed: {str(e2)}")
    except Exception as e:
        raise RuntimeError(f"Hirshfeld-I calculation failed: {str(e)}")

def get_dipole(rt_obj):
    rt_obj._dipole = rt_obj._scf.dip_moment(mol=rt_obj._scf.mol, dm=rt_obj.den_ao, unit='A.U.', verbose=1)

def _temp_get_dipole(rt_obj):
    # Temporary fix for argument name discrepancy in GHF.dip_moment ('unit_symbol' instead of 'unit')
    rt_obj._dipole = rt_obj._scf.dip_moment(mol=rt_obj._scf.mol, dm=rt_obj.den_ao, unit_symbol='A.U.', verbose=1)

def get_quadrupole(rt_obj):
    rt_obj._quadrupole = rt_obj._scf.quad_moment(mol=rt_obj._scf.mol, dm=rt_obj.den_ao,unit='A.U.', verbose=1)

def get_mag(rt_obj):
    Nsp = int(np.shape(rt_obj.ovlp)[0] / 2)

    magx = np.sum((rt_obj.den_ao[:Nsp, Nsp:] + rt_obj.den_ao[Nsp:, :Nsp]) * rt_obj.ovlp[:Nsp,:Nsp])
    magy = 1j * np.sum((rt_obj.den_ao[:Nsp, Nsp:] - rt_obj.den_ao[Nsp:, :Nsp]) * rt_obj.ovlp[:Nsp,:Nsp])
    magz = np.sum((rt_obj.den_ao[:Nsp, :Nsp] - rt_obj.den_ao[Nsp:, Nsp:]) * rt_obj.ovlp[:Nsp,:Nsp])
    rt_obj._mag = [magx, magy, magz]

def get_hirshfeld_mag(rt_obj):
    rho_aa, rho_ab, rho_ba, rho_bb = hirshfeld_partition(rt_obj._scf, rt_obj.den_ao, rt_obj.grids, rt_obj.atom_weights)
    mx = (rho_ab + rho_ba)
    my = 1j * (rho_ab - rho_ba)
    mz = (rho_aa - rho_bb)

    rt_obj._hirshfeld_mx_atoms = mx.sum(axis=1)
    rt_obj._hirshfeld_my_atoms = my.sum(axis=1)
    rt_obj._hirshfeld_mz_atoms = mz.sum(axis=1)

def get_mo_occ(rt_obj):
    # P_mo = C+SP_aoSC
    SP_aoS = np.matmul(rt_obj.ovlp,np.matmul(rt_obj.den_ao,rt_obj.ovlp))
    if rt_obj.nmat == 2:
        mo_coeff_print_transpose = np.stack((rt_obj.mo_coeff_print[0].T.conj(), rt_obj.mo_coeff_print[1].T.conj()))
        den_mo = np.matmul(mo_coeff_print_transpose,np.matmul(SP_aoS,rt_obj.mo_coeff_print))
        den_mo = np.real(np.sum(den_mo,axis=0))
    else:
        den_mo = np.matmul(rt_obj.mo_coeff_print.T.conj(), np.matmul(SP_aoS,rt_obj.mo_coeff_print))
        den_mo = np.real(den_mo)

    rt_obj._mo_occ = np.diagonal(den_mo)

def get_mo_occ_separate(rt_obj):
    # P_mo = C+SP_aoSC
    SP_aoS = np.matmul(rt_obj.ovlp,np.matmul(rt_obj.den_ao,rt_obj.ovlp))
    mo_coeff_print_transpose = np.stack((rt_obj.mo_coeff_print[0].T.conj(), rt_obj.mo_coeff_print[1].T.conj()))
    den_mo = np.matmul(mo_coeff_print_transpose,np.matmul(SP_aoS,rt_obj.mo_coeff_print))
    den_mo = np.real(den_mo)

    rt_obj._mo_occ_separate = [np.diagonal(den_mo[0]), np.diagonal(den_mo[1])]

def get_nuclei(rt_obj):
    rt_obj._nuclei = [rt_obj.nuc.labels, rt_obj.nuc.pos*lib.param.BOHR, rt_obj.nuc.vel*lib.param.BOHR, rt_obj.nuc.force]

def get_cube_density(rt_obj):
    '''
    Will create Gaussian cube file for molecule electron density
    for every propagation time given in rt_obj.cube_density_indices.
    '''
    if np.rint(rt_obj.current_time/rt_obj.timestep) in np.rint(np.array(rt_obj.cube_density_indices)/rt_obj.timestep):
        if hasattr(rt_obj, 'cube_filename'):
            cube_name = f'{rt_obj.cube_filename}{rt_obj.current_time}.cube'
        else:
            cube_name = f'{rt_obj.current_time}.cube'
        cubegen.density(rt_obj._scf.mol, cube_name, rt_obj.den_ao)

def get_spin_square(rt_obj):
    if rt_obj._scf.istype('UHF'):
        mo_coeff = (rt_obj._scf.mo_coeff[0][:,rt_obj.occ[0]>0],
                    rt_obj._scf.mo_coeff[1][:,rt_obj.occ[1]>0])
    else:
        mo_coeff = rt_obj._scf.mo_coeff[:,rt_obj.occ>0]

    rt_obj._s2, _ = rt_obj._scf.spin_square(mo_coeff)


def get_plane_partition_charge(rt_obj):
    frag1 = rt_obj.fragments[0]
    frag2 = rt_obj.fragments[1]
    coords = rt_obj._scf.mol.atom_coords()
    frag1_indices = frag1.match_indices
    frag2_indices = frag2.match_indices

    #here i want to compute the centers of mass, so the physical division is variable
    com1 = np.mean(coords[frag1_indices], axis=0)
    com2 = np.mean(coords[frag2_indices], axis=0)
    plane_origin = (com1 + com2) / 2
    plane_normal = com2 - com1
    plane_normal /= np.linalg.norm(plane_normal)

    #now i want to define the grid coordinates 
    grids = rt_obj.grids
    grid_coords = grids.coords
    rel_coords = grid_coords - plane_origin
    signed_dist = np.dot(rel_coords, plane_normal)

    frag1_mask = signed_dist < 0
    frag2_mask = signed_dist >= 0

    #now i have to get the density on both sides of the plane
    if rt_obj.nmat == 2:
        rho_a, rho_b = hirshfeld_partition(rt_obj._scf, rt_obj.den_ao, rt_obj.grids, rt_obj.atom_weights)
        rho = rho_a + rho_b
    elif rt_obj._scf.istype('GHF') | rt_obj._scf.istype('GKS'):
        rho_aa, rho_ab, rho_ba, rho_bb = hirshfeld_partition(rt_obj._scf, rt_obj.den_ao, rt_obj.grids, rt_obj.atom_weights)
        rho = rho_aa + rho_bb
    else:
        rho = hirshfeld_partition(rt_obj._scf, rt_obj.den_ao, rt_obj.grids, rt_obj.atom_weights)

    charge_frag1 = np.sum(rho[:, frag1_mask])
    charge_frag2 = np.sum(rho[:, frag2_mask])

    rt_obj._plane_partition_charge = [charge_frag1, charge_frag2]



def get_plane_partition_charge_spatial(rt_obj):
    if not hasattr(rt_obj, 'grids'):
        rt_obj.grids, _ = get_weights(rt_obj._scf.mol)
    grids = rt_obj.grids
    from pyscf.dft import numint
    ni = numint.NumInt()

    # Spin-summed total density
    if rt_obj.nmat == 2:
        rho_a = ni.get_rho(rt_obj._scf.mol, rt_obj.den_ao[0], grids)
        rho_b = ni.get_rho(rt_obj._scf.mol, rt_obj.den_ao[1], grids)
        rho_tot = rho_a + rho_b
    else:
        rho_tot = ni.get_rho(rt_obj._scf.mol, rt_obj.den_ao, grids)

    # Plane via fragment COMs
    frag1, frag2 = rt_obj.fragments[0], rt_obj.fragments[1]
    coords_atoms = rt_obj._scf.mol.atom_coords()
    com1 = coords_atoms[frag1.match_indices].mean(axis=0)
    com2 = coords_atoms[frag2.match_indices].mean(axis=0)
    plane_origin = 0.5 * (com1 + com2)
    normal = com2 - com1
    normal /= np.linalg.norm(normal)

    rel = grids.coords - plane_origin
    signed = rel @ normal
    mask1 = signed < 0
    mask2 = ~mask1

    w = grids.weights
    q1 = np.sum(rho_tot[mask1] * w[mask1])
    q2 = np.sum(rho_tot[mask2] * w[mask2])
    rt_obj._plane_partition_charge_spatial = [q1, q2]