"""
HORTON integration for TIDES: MBIS and Hirshfeld-I using PySCF densities.

This module evaluates the molecular electron density on HORTON Becke-Lebedev
grids using PySCF's AO basis and the provided AO density matrix(es), so the
basis and level of theory are inherited directly from the PySCF job.
"""

from typing import Dict, Optional, Tuple, Union, Callable
import numpy as np
import os
import tempfile
from pyscf import gto, scf, dft

# External dependencies
try:
    # HORTON 2.x
    from horton.grid import BeckeMolGrid, ExpRTransform, RadialGrid
    from horton.part import MBISWPart, HirshfeldIWPart, ProAtomDB
    _HORTON_AVAILABLE = True
except Exception:
    _HORTON_AVAILABLE = False

# PySCF
from pyscf.dft import numint as pyscf_numint


class HortonUnavailableError(RuntimeError):
    pass


def _ensure_horton():
    if not _HORTON_AVAILABLE:
        raise HortonUnavailableError(
            "The 'horton' library is not available. Install HORTON 2.x to use MBIS/Hirshfeld-I."
        )


def _build_becke_grid(
    coordinates: np.ndarray,
    numbers: np.ndarray,
    pseudo_numbers: Optional[np.ndarray] = None,
    lebedev_order: int = 110,
    rgrid_params: Tuple[float, float, int] = (5e-4, 2e1, 120),
    local: bool = True,
) -> "BeckeMolGrid":
    """
    Construct a Becke molecular grid as used by HORTON partitioning.
    """
    _ensure_horton()
    if pseudo_numbers is None:
        pseudo_numbers = numbers
    r0, rmax, nr = rgrid_params
    rtf = ExpRTransform(r0, rmax, nr)
    rgrid = RadialGrid(rtf)
    mode = "only" if local else "discard"
    return BeckeMolGrid(coordinates, numbers, pseudo_numbers, (rgrid, lebedev_order),
                        random_rotate=False, mode=mode)


def _make_density_functions_from_pyscf(
    scf_obj,
    den_ao: np.ndarray,
) -> Tuple[Callable[[np.ndarray], np.ndarray], Optional[Callable[[np.ndarray], np.ndarray]]]:
    """
    Create callables to evaluate total density (and optional spin density) at arbitrary points.

    Returns:
      rho_fn(points) -> (npts,)
      s_fn(points) -> (npts,) or None
    """
    ni = pyscf_numint.NumInt()
    mol = scf_obj.mol

    def _eval_rho_from_dm(points: np.ndarray, dm: np.ndarray) -> np.ndarray:
        ao = ni.eval_ao(mol, points)
        # rho = sum_{mu,nu} AO_mu * DM_{mu,nu} * AO_nu*
        rho = np.einsum('pi,ij,pj->p', ao, dm, ao.conjugate(), optimize=True)
        return np.real(rho)

    # UKS/UHF: two density matrices
    if hasattr(scf_obj, "istype") and (scf_obj.istype('UHF') or scf_obj.istype('UKS')):
        dm_a = den_ao[0]
        dm_b = den_ao[1]

        def rho_fn(points: np.ndarray) -> np.ndarray:
            return _eval_rho_from_dm(points, dm_a) + _eval_rho_from_dm(points, dm_b)

        def s_fn(points: np.ndarray) -> np.ndarray:
            return _eval_rho_from_dm(points, dm_a) - _eval_rho_from_dm(points, dm_b)

        return rho_fn, s_fn

    # GHF/GKS: block 2x2 spin density matrix
    if hasattr(scf_obj, "istype") and (scf_obj.istype('GHF') or scf_obj.istype('GKS')):
        nao = mol.nao
        dm = den_ao
        dm_aa = dm[:nao, :nao]
        dm_bb = dm[nao:, nao:]

        def rho_fn(points: np.ndarray) -> np.ndarray:
            return _eval_rho_from_dm(points, dm_aa) + _eval_rho_from_dm(points, dm_bb)

        def s_fn(points: np.ndarray) -> np.ndarray:
            return _eval_rho_from_dm(points, dm_aa) - _eval_rho_from_dm(points, dm_bb)

        return rho_fn, s_fn

    # RHF/RKS (single matrix)
    def rho_fn(points: np.ndarray) -> np.ndarray:
        return _eval_rho_from_dm(points, den_ao)

    return rho_fn, None


def compute_mbis_from_pyscf(
    scf_obj,
    den_ao: np.ndarray,
    *,
    lebedev_order: int = 110,
    rgrid_params: Tuple[float, float, int] = (5e-4, 2e1, 120),
    local_grid: bool = True,
    lmax: int = 3,
    threshold: float = 1e-8,
    maxiter: int = 200,
) -> Dict[str, np.ndarray]:
    """
    Run HORTON MBIS using densities evaluated from PySCF AO density matrix.

    Returns:
      dict with keys: charges, valence_charges, core_charges, valence_widths
    """
    _ensure_horton()
    mol = scf_obj.mol
    coordinates = mol.atom_coords(unit='Bohr')
    numbers = mol.atom_charges()
    pseudo_numbers = numbers

    grid = _build_becke_grid(
        coordinates, numbers, pseudo_numbers,
        lebedev_order=lebedev_order,
        rgrid_params=rgrid_params,
        local=local_grid,
    )

    rho_fn, s_fn = _make_density_functions_from_pyscf(scf_obj, den_ao)
    rho = rho_fn(grid.points)
    spindens = s_fn(grid.points) if s_fn is not None else None

    wpart = MBISWPart(
        coordinates, numbers, pseudo_numbers,
        grid, rho,
        spindens=spindens, local=local_grid, lmax=lmax, threshold=threshold, maxiter=maxiter,
    )
    wpart.do_charges()

    out = {
        "charges": wpart["charges"].copy(),
        "valence_charges": wpart["valence_charges"].copy(),
        "core_charges": wpart["core_charges"].copy(),
        "valence_widths": wpart["valence_widths"].copy(),
    }
    return out


def compute_hirshfeld_i_from_pyscf(
    scf_obj,
    den_ao: np.ndarray,
    *,
    proatomdb: Union[str, "ProAtomDB"],
    lebedev_order: int = 110,
    rgrid_params: Tuple[float, float, int] = (5e-4, 2e1, 120),
    local_grid: bool = True,
    lmax: int = 3,
    threshold: float = 1e-6,
    maxiter: int = 500,
) -> Dict[str, np.ndarray]:
    """
    Run HORTON Hirshfeld-I using densities evaluated from PySCF AO density matrix.

    proatomdb: path to atoms.h5 or a ProAtomDB instance (required for Hirshfeld(-I)).
    """
    _ensure_horton()
    if isinstance(proatomdb, str):
        padb = ProAtomDB.from_file(proatomdb)
    else:
        padb = proatomdb

    mol = scf_obj.mol
    coordinates = mol.atom_coords(unit='Bohr')
    numbers = mol.atom_charges()
    pseudo_numbers = numbers

    grid = _build_becke_grid(
        coordinates, numbers, pseudo_numbers,
        lebedev_order=lebedev_order,
        rgrid_params=rgrid_params,
        local=local_grid,
    )

    rho_fn, s_fn = _make_density_functions_from_pyscf(scf_obj, den_ao)
    rho = rho_fn(grid.points)
    spindens = s_fn(grid.points) if s_fn is not None else None

    wpart = HirshfeldIWPart(
        coordinates, numbers, pseudo_numbers,
        grid, rho,
        padb,
        spindens=spindens, local=local_grid, lmax=lmax, threshold=threshold, maxiter=maxiter,
    )
    wpart.do_charges()

    return {"charges": wpart["charges"].copy()}

    # Add this function to your horton_part.py file
# Replace your create_proatomdb_from_pyscf function with this corrected version
def create_proatomdb_from_pyscf(mol, method='dft', xc='pbe', basis=None, filename=None):
    """
    Create a HORTON proatom database from PySCF molecule.
    """
    _ensure_horton()
    from horton.part import ProAtomDB, ProAtomRecord
    import tempfile
    import os
    
    # Get unique atom types in the molecule
    unique_atoms = set([mol.atom_symbol(i) for i in range(mol.natm)])
    
    # Create a database
    if filename is None:
        # Create a temporary directory to store the database
        temp_dir = tempfile.mkdtemp(prefix="horton_proatoms_")
        filename = os.path.join(temp_dir, "atoms.h5")
    
    # Initialize list to hold all records
    all_records = []
    
    # For each unique atom type, compute all possible charged states
    for symbol in unique_atoms:
        pure_symbol = ''.join(c for c in symbol if not c.isdigit())
        atomic_number = {s: i for i, s in enumerate(
            ['X', 'H', 'He', 'Li', 'Be', 'B', 'C', 'N', 'O', 'F', 'Ne', 
             'Na', 'Mg', 'Al', 'Si', 'P', 'S', 'Cl', 'Ar', 'K', 'Ca'])}[pure_symbol]
        
        # Extract the proper basis for this element from the molecule
        # If basis is a dict, find the right basis for this atom
        if basis is None:
            # If the basis is in the mol.basis dictionary
            if isinstance(mol.basis, dict):
                # Try to find the specific basis for this atom type 
                # (like 'H1', 'H2', 'O1', 'O2')
                atom_basis = None
                for key in mol.basis:
                    if pure_symbol in key:
                        atom_basis = mol.basis[key]
                        break
                
                # If no specific basis found, use default
                if atom_basis is None:
                    atom_basis = 'sto-3g'  # Fallback to a simple basis
            else:
                # If mol.basis is a string, use that
                atom_basis = mol.basis
        else:
            # Use the provided basis
            atom_basis = basis
            
        # Loop over all reasonable charge states
        max_charge = min(atomic_number, 2)  # Maximum reasonable charge
        for charge in range(-2, max_charge + 1):
            # Skip unreasonable charge states
            if atomic_number + charge < 0:
                continue
                
            # Create an atom
            atom = gto.Mole()
            atom.atom = f"{pure_symbol} 0 0 0"  # Use pure symbol without digits
            atom.basis = atom_basis
            atom.charge = charge
            atom.spin = (atomic_number + charge) % 2  # Open or closed shell
            atom.verbose = 0
            
            try:
                atom.build()
                
                # Run the calculation
                if method.lower() == 'dft':
                    atm_scf = dft.RKS(atom)
                    atm_scf.xc = xc
                else:
                    atm_scf = scf.RHF(atom)
                    
                energy = atm_scf.kernel()
                
                # Create radial grid for the pro-atom
                r0, rmax, nr = (1e-5, 2e1, 200)  # Radial grid parameters
                rtf = ExpRTransform(r0, rmax, nr)
                rgrid = RadialGrid(rtf)
                
                # Create a proper spherical grid
                # Use multiple angles for each radius to avoid node problems
                r_points = np.linspace(1e-5, rmax, nr)
                rho_values = np.zeros(nr)
                
                # For each radius, sample multiple points and average
                for i, r in enumerate(r_points):
                    # Create points at this radius with different angles
                    theta = np.linspace(0, np.pi, 5)
                    phi = np.linspace(0, 2*np.pi, 5)
                    
                    points = []
                    for t in theta:
                        for p in phi:
                            x = r * np.sin(t) * np.cos(p)
                            y = r * np.sin(t) * np.sin(p)
                            z = r * np.cos(t)
                            points.append([x, y, z])
                    
                    points = np.array(points)
                    
                    # Evaluate density on these points
                    ao = pyscf_numint.NumInt().eval_ao(atom, points)
                    dm = atm_scf.make_rdm1()
                    try:
                        # Try direct contraction first
                        rho_vals = np.einsum('pi,ij,pj->p', ao, dm, ao.conjugate(), optimize=True).real
                        # Average over all angles at this radius
                        rho_values[i] = np.mean(rho_vals)
                    except Exception as e1:
                        try:
                            # Try manual contraction
                            rho_vals = np.zeros(len(points))
                            for p in range(len(points)):
                                ao_p = ao[p]
                                rho_vals[p] = np.sum(ao_p.reshape(-1, 1) * dm * ao_p.conjugate().reshape(1, -1)).real
                            rho_values[i] = np.mean(rho_vals)
                        except Exception as e2:
                            # Last resort: use a constant small value
                            print(f"Warning: Density evaluation failed at r={r}: {str(e1)}, {str(e2)}")
                            rho_values[i] = 1e-10 + atomic_number * np.exp(-r)
                
                # Create a ProAtomRecord with the spherically averaged density
                record = ProAtomRecord(
                    number=atomic_number,
                    charge=charge,
                    energy=energy,
                    rgrid=rgrid,
                    rho=rho_values
                )
                
                # Add to our records list
                all_records.append(record)
                
                print(f"Added {pure_symbol} with charge {charge} to proatom database")
            except Exception as e:
                print(f"Warning: Failed to converge {pure_symbol} with charge {charge}: {str(e)}")
    
    # Create the database with all records
    padb = ProAtomDB(all_records)
    
    # Save the database
    if filename:
        padb.to_file(filename)
        print(f"ProAtomDB saved to {filename}")
    
    return padb