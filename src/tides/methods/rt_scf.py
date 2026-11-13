import numpy as np
from scipy.linalg import inv
from pyscf import gto
from pyscf.scf import addons
from pyscf.lib import logger
from tides.propagators.rt_prop import propagate
from tides.observables import rt_observables
from tides.observables import rt_output
from tides.utils.rt_utils import restart_from_chkfile
import os
import warnings


'''
Real-time SCF Class
'''

def _orth_inverse(orth):
    orth = np.asarray(orth)
    if orth.shape[-1] != orth.shape[-2]:
        raise ValueError(
            f'The orthogonalisation matrix is {orth.shape[-2]}x{orth.shape[-1]}: the AO basis is (near-)linearly '
            f'dependent and {orth.shape[-2] - orth.shape[-1]} combination(s) were dropped by canonical '
            'orthogonalisation. TiDES needs a square orthogonaliser: remove the offending diffuse functions, '
            'or pass a square `orth` (e.g. S^-1/2).')
    return inv(orth)

class RT_SCF:
    def __init__(self, scf, timestep, max_time, filename=None, prop=None, frequency=1, orth=None, chkfile='tides.chk', verbose=3):

        self.timestep = timestep
        self.frequency = frequency
        self.max_time = max_time
        self._scf = scf
        self.ovlp = self._scf.get_ovlp()
        # Use the occupations the SCF object holds (keeps non-Aufbau / Delta-SCF references);
        # fall back to Aufbau only if there are none. Copy: excite() edits self.occ in place.
        mo_occ = getattr(self._scf, 'mo_occ', None)
        self.occ = np.array(mo_occ, copy=True) if mo_occ is not None else self._scf.get_occ()

        self.verbose = verbose
        self._potential = []
        self.fragments = []

        self.labels = [self._scf.mol._atom[idx][0] for idx, _ in enumerate(self._scf.mol._atom)]
        if prop is None: prop = 'magnus_interpol'
        if orth is None: orth = addons.canonical_orth_(self.ovlp)
        self.prop = prop

        self.orth = orth  # the property setter also refreshes the cached inverse, self.orth_inv

        if filename is None:
            self._log = logger.Logger(verbose=self.verbose)
        else:
            #self._fh = open(filename, 'w')
            self._fh = open(filename, 'a') # Temporarily making _fh append to file
            self._log = logger.Logger(self._fh, verbose=self.verbose)

        self.den_ao = self._scf.make_rdm1(mo_occ=self.occ)
        if len(np.shape(self.den_ao)) == 3:
            self.nmat = 2
        else:
            self.nmat = 1

        # Restart from chkfile, or create a chkfile
        # If restarting from chkfile, self.den_ao will be rewritten
        # Pass chkfile=None explicitly to disable chkfile use entirely
        self.chkfile = chkfile
        if self.chkfile is not None and os.path.exists(self.chkfile):
            shape_before = np.shape(self._scf.mo_coeff)
            restart_from_chkfile(self)
            if np.shape(self._scf.mo_coeff) != shape_before:
                raise ValueError(
                    f'chkfile {self.chkfile} holds MO coefficients of shape {np.shape(self._scf.mo_coeff)}, but this '
                    f'calculation has shape {shape_before}. It belongs to a different calculation: delete it, or '
                    'pass another chkfile (chkfile=None disables checkpointing).')
            self.den_ao = self._scf.make_rdm1(mo_occ=self.occ)
            if round(self.max_time / self.timestep) <= round(self.current_time / self.timestep):
                msg = (f'Restarted from {self.chkfile} at t = {self.current_time} au, which is at or past '
                       f'max_time = {self.max_time} au: no steps will be taken. If this is meant to be a new '
                       'calculation, delete the chkfile or pass a different one (chkfile=None disables it).')
                self._log.warn(msg)
                warnings.warn(msg)
        else:
            self.current_time = 0
        self._t0 = self.current_time

        rt_observables._init_observables(self)

    # The orthogonaliser and its inverse must always change together.
    @property
    def orth(self):
        return self._orth

    @orth.setter
    def orth(self, value):
        self._orth = value
        self.orth_inv = _orth_inverse(value)

    def istype(self, type_code):
        if isinstance(type_code, type):
            return isinstance(self, type_code)

        return any(type_code == t.__name__ for t in self.__class__.__mro__)

    def update_time(self):
        self.current_time += self.timestep

    def get_fock_orth(self, den_ao):
        self.fock_ao = self._scf.get_fock(dm=den_ao).astype(np.complex128)
        if self._potential: self.apply_potential()
        return np.matmul(self.orth.conj().T, np.matmul(self.fock_ao, self.orth))

    def rotate_coeff_to_orth(self, coeff_ao):
        return np.matmul(self.orth_inv, coeff_ao)

    def rotate_fock_to_orth(self, fock_ao):
        return np.matmul(self.orth.T, np.matmul(fock_ao, self.orth))

    def rotate_coeff_to_ao(self, coeff_orth):
        return np.matmul(self.orth, coeff_orth)

    def add_potential(self, *args):
        for v_ext in args:
            self._potential.append(v_ext)

    def apply_potential(self):
        for v_ext in self._potential:
            self.fock_ao += v_ext.calculate_potential(self)

    def kernel(self, mo_coeff_print=None):
        try:
            propagate(self, mo_coeff_print)
        except Exception:
            raise
        finally:
            #if np.isclose(self.current_time, self.max_time + self._t0):
            if np.isclose(self.current_time, self.max_time): # So calculation terminates once max_time is reached after restarts
                self._log.note('Done')
            else:
                self._log.note('Propagation Stopped Early')
            if hasattr(self, 'fh'):
                self.fh.close()
            if hasattr(self, '_xyz_fh'):
                self._xyz_fh.close()

        return self
