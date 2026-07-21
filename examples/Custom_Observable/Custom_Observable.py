from pyscf import gto, scf
from tides import rt_scf

'''
This example walks through how to implement a custom observable. 
We'll re-define the dipole moment (even though it's already defined in TiDES) as an example.

This illustrates that new observables can be defined from the input file without having to deal with modifying the source code of TiDES.
'''

# Build mol
h2o_mol = gto.M(atom='''
  O     0.00000000    -0.00001441    -0.34824012
  H    -0.00000000     0.76001092    -0.93285191
  H     0.00000000    -0.75999650    -0.93290797
        ''', basis='6-31G')

# Run RHF for water
h2o = scf.RHF(h2o_mol)
h2o.kernel()

# Create RT_SCF object
rt_h2o = rt_scf.RT_SCF(h2o, 0.2, 10, filename=None, prop='magnus_interpol', frequency=1, orth=None, chkfile=None, verbose=3)

# Redefine dipole observable
def get_redefined_dipole(rt_scf, den_ao):
    # PySCF's SCF objects have a dip_moment() method that returns the dipole moment.
    rt_scf._redefined_dipole = rt_scf._scf.dip_moment(mol=rt_scf._scf.mol, dm=rt_scf.den_ao, unit='A.U.', verbose=1)

def print_redefined_dipole(rt_scf):
    # Format output however you want
    rt_scf._log.note(f'HERE IS THE CUSTOM DIPOLE OBSERVABLE (AU): {" ".join(map(str,rt_scf._redefined_dipole))}')


# Add these functions to rt_h2o._observables_functions dictionary and declare the custom observable in rt_h2o.observables dictionary.
# Make sure to keep the key name consistent

rt_h2o._observables_functions['redefined_dipole'] = [get_redefined_dipole, print_redefined_dipole]

rt_h2o.observables['redefined_dipole'] = True

rt_h2o.kernel()
