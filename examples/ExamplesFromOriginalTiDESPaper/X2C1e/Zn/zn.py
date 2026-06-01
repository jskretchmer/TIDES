import numpy as np
from pyscf import gto, scf, dft, lib
from tides import rt_scf
import scipy
from sapporo import sapporo

mol = gto.M(
	verbose = 0,
	atom='''
	Zn 0 0 0  
''',
	basis=sapporo)

zn = dft.GKS(mol).x2c()
zn.xc = 'Slater, VWN5'
zn.kernel()

rt_zn = rt_scf.RT_SCF(zn, 0.1, 10000)
rt_zn.observables.update(charge=True, dipole=True)

class EfieldWithX2C:
    def __init__(self, mf, amplitude, center=0):
        self.amplitude = np.array(amplitude)
        self.center = center
        self.get_tdip(mf)

    def calculate_field_energy(self, rt_mf):
        if rt_mf.current_time == self.center:
            return self.amplitude
        else:
            return np.array([0, 0, 0])

    def _block_diag(self, mat):
        '''
        [A 0]
        [0 A]
        '''
        return scipy.linalg.block_diag(mat, mat)

    def _sigma_dot(self, mat):
        '''sigma dot A x B + A dot B'''
        quaternion = np.vstack([1j * lib.PauliMatrices, np.eye(2)[None,:,:]])
        nao = mat.shape[-1] * 2
        return lib.einsum('sxy,spq->xpyq', quaternion, mat).reshape(nao, nao)

    def get_tdip(self, mf):
        xmol = mf.with_x2c.get_xmol()[0]
        nao = xmol.nao
        r = xmol.intor_symmetric('int1e_r')
        r = np.array([self._block_diag(x) for x in r])
        c1 = 0.5/lib.param.LIGHT_SPEED
        prp = xmol.intor_symmetric('int1e_sprsp').reshape(3,4,nao,nao)
        prp = np.array([self._sigma_dot(x*c1**2) for x in prp])
        self.tdip = -1 * mf.with_x2c.picture_change((r, prp))

    def calculate_potential(self, rt_mf):
        energy = self.calculate_field_energy(rt_mf)

        dipole_term = -1 * np.einsum('xij,x->ij', self.tdip, energy)
        return dipole_term

delta_field = EfieldWithX2C(zn, [0.0001, 0.0000, 0.0000])

rt_zn.add_potential(delta_field)

rt_zn.kernel()

