from pyscf import gto, dft, md
import numpy as np

mol_ref = gto.M(
    verbose = 0,
    atom='''
Cl           0.00000000 0.00000000 0.00000000
''', basis='6-31G*', spin=1)
nao = mol_ref.nao
cl2_ref = dft.UKS(mol_ref); cl2_ref.xc = 'B3LYP'
cl2_ref.kernel()
dm_ref = np.copy(cl2_ref.make_rdm1())
dm_guess = np.zeros((2,2*nao,2*nao))
dm_guess[0,:nao,:nao] = np.copy(dm_ref[0,:,:]); dm_guess[0,nao:,nao:] = np.copy(dm_ref[1,:,:])
dm_guess[1,:nao,:nao] = np.copy(dm_ref[1,:,:]); dm_guess[1,nao:,nao:] = np.copy(dm_ref[0,:,:])

class UKS_asym(dft.uks.UKS):
    def kernel(self, dm0=None, **kwargs):
        udm_guess = np.copy(dm_guess)
        result = super().kernel(dm0=udm_guess, **kwargs)
        return result

# Build Cl2 molecule
mol = gto.M(
    verbose = 0,
    atom='''
Cl           0.00000000 0.00000000 0.00000000
Cl           0.00000000 0.00000000 2.00000000
''',
    basis='6-31G*', spin=0)

# Build Unrestricted Kohn-Sham object
cl2 = UKS_asym(mol)
cl2.xc = 'B3LYP'
cl2.kernel()

# Set initial velocity
init_eV = 3.0
init_velo = np.sqrt((init_eV/(2*27.2114))*2/63744)

myscanner = cl2.nuc_grad_method().as_scanner()

myintegrator = md.NVE(myscanner,
                            dt=0.5,
                            steps=10000,
                            veloc=np.array([[0.0, 0.0, -1*init_velo],
                                           [0.0, 0.0, init_velo]]),
                            data_output="BOMD.md.data",
                            trajectory_output="BOMD.md.xyz").run()

myintegrator.data_output.close()
myintegrator.trajectory_output.close()

