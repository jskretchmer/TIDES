from pyscf import gto, scf
from tides import rt_scf

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

# Define blank custom observable
def get_custom_observable(rt_scf, den_ao):
    # rt_scf and den_ao are required arguments
    rt_scf._custom_observable = None

def print_custom_observable(rt_scf):
    # Format output however you want
    rt_scf._log.note(f'HERE IS THE CUSTOM OBSERVABLE: {rt_scf._custom_observable}')


# Add these functions to rt_h2o._observables_functions dictionary and declare the custom observable in rt_h2o.observables dictionary.
# Make sure to keep the key name consistent

rt_h2o._observables_functions['custom'] = [get_custom_observable, print_custom_observable]

rt_h2o.observables['custom'] = True

rt_h2o.kernel()
