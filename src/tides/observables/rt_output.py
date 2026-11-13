import numpy as np
import sys
np.set_printoptions(threshold=sys.maxsize, linewidth=sys.maxsize)

'''
Real-time Output Functions
'''

def update_output(rt_obj):
    rt_obj._log.note(f'{"="*25} \n')
    rt_obj._log.note(f'Current Time (AU): {rt_obj.current_time:.8f} \n')
    for key, function in rt_obj._observables_functions.items():
        function[1](rt_obj)

    rt_obj._log.note(f'{"="*25} \n')

def _print_energy(rt_obj):
    energy = rt_obj._energy
    rt_obj._log.note(f'Total Energy (AU): {energy[0]} \n')
    if len(energy) > 1:
        for index, fragment in enumerate(energy[1:]):
            rt_obj._log.note(f'Fragment {index + 1} Energy (AU): {fragment} \n')
    if rt_obj.istype('RT_Ehrenfest'):
        kinetic_energy = rt_obj._kinetic_energy
        rt_obj._log.note(f'Total Kinetic Energy (AU): {np.sum(kinetic_energy)} \n')
        rt_obj._log.info(f'Atom Kinetic Energies (AU):')
        for atom in zip(rt_obj.nuc.labels, kinetic_energy):
            rt_obj._log.info(f' {atom[0]} {atom[1]}')
        rt_obj._log.info(' ')
        for index, frag in enumerate(rt_obj.fragments):
            rt_obj._log.note(f'Fragment {index + 1} Kinetic Energy (AU): {np.sum(kinetic_energy[frag.match_indices])} \n')

def _print_ao_occ(rt_obj):
    ao_occ = rt_obj._ao_occ
    rt_obj._log.note(f'Atomic Orbital Occupations: {" ".join(map(str,ao_occ))} \n')

def _print_mo_occ(rt_obj):
    mo_occ = rt_obj._mo_occ
    rt_obj._log.note(f'Molecular Orbital Occupations: {" ".join(map(str,mo_occ))} \n')

def _print_mo_occ_separate(rt_obj):
    mo_occ_separate = rt_obj._mo_occ_separate
    rt_obj._log.note(f'Molecular Orbital Alpha Occupations: {" ".join(map(str,mo_occ_separate[0]))} \n')
    rt_obj._log.note(f'Molecular Orbital Beta Occupations: {" ".join(map(str,mo_occ_separate[1]))} \n')

def _print_charge(rt_obj):
    charge = rt_obj._charge
    rt_obj._log.note(f'Total Electronic Charge: {np.real(charge[0])} \n')
    if len(charge) > 1:
        for index, fragment in enumerate(charge[1:]):
            rt_obj._log.note(f'Fragment {index + 1} Electronic Charge: {np.real(fragment)} \n')

def _print_hirshfeld_charge(rt_obj):
    labels = rt_obj.labels
    atom_charges = rt_obj._hirshfeld_charges
    rt_obj._log.note('Hirshfeld Atomic Electronic Charges:')
    for atom in zip(labels, atom_charges):
        rt_obj._log.note(f' {atom[0]} \t {np.real(atom[1])}')
    rt_obj._log.note(' ')

def _print_dipole(rt_obj):
    dipole = rt_obj._dipole
    rt_obj._log.note(f'Total Dipole Moment [X, Y, Z] (AU): {" ".join(map(str,dipole))} \n')

def _print_quadrupole(rt_obj):
    quadrupole = rt_obj._quadrupole
    rt_obj._log.note(f'Total Quadrupole Moment [[XX,XY,XZ], [YX,YY,YZ], [ZX,ZY,ZZ]] (AU): {" ".join(map(str,quadrupole))} \n')

def _print_mag(rt_obj):
    mag = rt_obj._mag
    rt_obj._log.note(f'Total Magnetization [X, Y, Z]: {" ".join(map(str,np.real(mag)))} \n')

def _print_hirshfeld_mag(rt_obj):
    labels = rt_obj.labels
    mx = rt_obj._hirshfeld_mx_atoms
    my = rt_obj._hirshfeld_my_atoms
    mz = rt_obj._hirshfeld_mz_atoms
    m = np.transpose([mx, my, mz])
    rt_obj._log.note(f'Hirshfeld Magnetization [X, Y, Z]:')
    for atom in zip(labels, m):
        rt_obj._log.note(f' {atom[0]}: {np.real(atom[1][0])} {np.real(atom[1][1])} {np.real(atom[1][2])}')
    rt_obj._log.note(' ')

def _print_mulliken_charge(rt_obj):
    labels = rt_obj.labels
    atom_charges = rt_obj._atom_charges
    rt_obj._log.note('Atomic Electronic Charges:')
    for atom in zip(labels, atom_charges):
        rt_obj._log.note(f' {atom[0]} \t {np.real(atom[1])}')
    rt_obj._log.note(' ')

def _print_nuclei(rt_obj):
    rt_obj._xyz_log.note(f'{rt_obj._scf.mol.natm}')
    rt_obj._xyz_log.note(f'Current Time (AU): {rt_obj.current_time:.8f}')
    rt_obj._update_xyz(rt_obj, rt_obj._nuclei)

def _nuclei_coords(rt_obj, nuclei):
    for atom in zip(nuclei[0], nuclei[1]):
        coords_str = "\t".join([f"{x:.11f}" for x in atom[1]])
        rt_obj._xyz_log.note(f'{atom[0]} \t {coords_str}')
        #rt_obj._xyz_log.note(f'{atom[0]} \t {"\t".join(map(lambda x: f"{x:.11f}",atom[1]))}')

def _nuclei_coords_vels(rt_obj, nuclei):
    for atom in zip(nuclei[0], nuclei[1], nuclei[2]):
        coords_str = "\t".join([f"{x:.11f}" for x in atom[1]])
        vels_str   = "\t".join([f"{x:.11f}" for x in atom[2]])
        rt_obj._xyz_log.note(f'{atom[0]} \t {coords_str} \t {vels_str}')

def _nuclei_coords_vels_forces(rt_obj, nuclei):
    for atom in zip(nuclei[0], nuclei[1], nuclei[2], nuclei[3]):
        coords_str  = "\t".join([f"{x:.11f}" for x in atom[1]])
        vels_str    = "\t".join([f"{x:.11f}" for x in atom[2]])
        forces_str  = "\t".join([f"{x:.11f}" for x in atom[3]])
        rt_obj._xyz_log.note(f'{atom[0]} \t {coords_str} \t {vels_str} \t {forces_str}')

def _print_spin_square(rt_obj):
    s2 = rt_obj._s2
    _2s_p1 = rt_obj._2s_p1
    rt_obj._log.note(f'S^2: {s2}')
    rt_obj._log.note(f'2S+1: {_2s_p1} \n')

def _print_mo_coeff(rt_obj):
    rt_obj._log.note(f'\n{"*"*25} Molecular Orbital Coefficients (AO Basis): {"*"*25}\n {rt_obj._scf.mo_coeff} \n{"*"*50}\n')

def _print_den_ao(rt_obj):
    rt_obj._log.note(f'\n{"@"*25} Density Matrix (AO Basis): {"@"*25}\n {rt_obj.den_ao} \n{"@"*50}\n')

def _print_fock_ao(rt_obj):
    rt_obj._log.note(f'\n{"+"*25} Fock Matrix (AO Basis): {"+"*25}\n {rt_obj.fock_ao} \n{"+"*50}\n')

def _print_civec(rt_obj):
    rt_obj._log.note(f'\n{"+"*25} CI Vector: {"+"*25}\n {rt_obj._scf.ci} \n{"+"*50}\n')

def _print_plane_partition_charge(rt_scf):
    charge = rt_scf._plane_partition_charge
    rt_scf._log.note(f'Plane Partition Charges: Frag1={charge[0]:.6f}, Frag2={charge[1]:.6f}\n')

def _print_plane_partition_charge_spatial(rt_obj):
    charge = rt_obj._plane_partition_charge_spatial
    rt_obj._log.note(f'Plane Partition Charges (Spatial Integration): Frag1={charge[0]:.6f}, Frag2={charge[1]:.6f}\n')