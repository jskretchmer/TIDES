"""
Note on X2C
When running

>>> mol.GHF().x2c()

Pyscf produces an X2C calcultion in the spherical GTO basis.
For RHF and UHF,

>>> mol.UHF().x2c()

Pyscf produces a scalar relativistic calculation with the X2C hamiltonian.
Running in the j-adapted spinor basis should converge to the same solution
in the spin-orbital real GTO basis. If you want to run in the spinor basis,

>>> mol.GHF().x2c1e()

this may have undesirable effects in TIDES.
"""

import numpy as np
from pyscf import gto, scf, x2c
from tides import rt_scf

frequency = 0.0428 # 1048nm laser
end_time  = 750

mol = gto.M(
    verbose = 0,
    atom='''
 C     2.082055   0.000019   -0.005536 
 C     1.363256   1.214579   -0.005000 
 C     -0.021952  1.215563   -0.003939 
 C     -0.710450  -0.000019  -0.002698 
 C     -0.021948  -1.215631  -0.003744 
 C     1.363238   -1.214592  -0.004823 
 H     1.904583   2.157205   -0.009568 
 H     -0.583166  2.141979   -0.001405 
 N     -2.166225  0.000009   0.002996
 H     -0.583092  -2.142083  -0.001120 
 H     1.904589   -2.157204  -0.009229 
 N     3.461411   0.000012   -0.055505 
 H     3.935742   0.846126   0.228448  
 H     3.935677   -0.846109  0.228537  
 O     -2.744155  -1.090769  0.005160  
 O     -2.744074  1.090823   0.005633  
  ''',
    basis='sto6g',
    spin = 0,
)

class cosfieldX2C:
    """Oscillating E-field for X2C"""
    def __init__(self, rt_scf, amplitude, frequency):
        self.amplitude = np.array(amplitude)
        self.frequency = frequency
        mol = rt_scf._scf.mol
        charges = mol.atom_charges()
        coords = mol.atom_coords()
        nuc_charge_center = np.einsum('z,zx->x', charges, coords) / charges.sum()
        mol.set_common_orig_(nuc_charge_center)

        self.tdip = rt_scf._scf.with_x2c.picture_change(('int1e_r_spinor', 'int1e_sprsp_spinor'))

    def calculate_potential(self, rt_scf):
        self.field = np.cos(self.frequency * rt_scf.current_time)
        # add the linear ramp for the first period
        if (rt_scf.current_time < 2 * np.pi / self.frequency):
            self.field *= rt_scf.current_time * self.frequency / (2 * np.pi)
        return -1 * np.einsum('xij,x->ij', self.tdip, self.field*self.amplitude)

# This part isn't necessary, but I like to see the progress
from tides import rt_output
def _custom_dipole(rt_scf):
    dipole = rt_scf._dipole
    rt_scf._log.note(f'Total Dipole Moment [X, Y, Z] (AU): {" ".join(map(str,dipole))} \n')
    c_time = rt_scf.current_time
    print(f"{fn}:  t = {c_time:.1f}   {int(100*c_time/end_time)}%")

# hijack the dipole printer :)
rt_output._print_dipole = _custom_dipole

fields = {
    "x_pos": [ .002, 0, 0],
    "x_neg": [-.002, 0, 0],
    "y_pos": [ 0, .002, 0],
    "y_neg": [ 0,-.002, 0],
    "z_pos": [ 0, 0, .002],
    "z_neg": [ 0, 0,-.002],
}

for fn, amp in fields.items():
    _scf = mol.GHF().x2c()
    _scf.kernel()

    rt_mf = rt_scf.RT_SCF(_scf, dt, end_time, filename=fn)
    rt_mf.observables.update(energy=True, dipole=True)
    rt_mf.add_potential(cosfieldX2C(rt_mf, amp, frequency))
    rt_mf.kernel()

