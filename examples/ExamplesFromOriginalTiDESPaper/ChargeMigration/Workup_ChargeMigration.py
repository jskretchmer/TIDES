import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt
from matplotlib import font_manager
from matplotlib.ticker import MultipleLocator, FormatStrFormatter, MaxNLocator
import matplotlib.patches as mpatches
from tides.parse_rt import parse_output
mpl.rcParams['pdf.fonttype'] = 42
mpl.rcParams['ps.fonttype'] = 42
mpl.rcParams['font.family'] = 'sans-serif'
mpl.rcParams['axes.linewidth'] = 1.0
mpl.rcParams['xtick.major.size'] = 5
mpl.rcParams['xtick.major.width'] = 0.75
mpl.rcParams['xtick.minor.size'] = 3
mpl.rcParams['xtick.minor.width'] = 0.5
mpl.rcParams['ytick.major.size'] = 5
mpl.rcParams['ytick.major.width'] = 0.75
mpl.rcParams['ytick.minor.size'] = 3
mpl.rcParams['ytick.minor.width'] = 0.5
mpl.rcParams['legend.facecolor'] = 'None'
mpl.rcParams['legend.edgecolor'] = 'k'

from pyscf import gto
from pyscf.tools import cubegen

def get_hole_density(result, idx):

    alpha_coeff = np.loadtxt('NOSCF_ORBITALS_ALPHA.txt', dtype=np.complex128)
    beta_coeff = np.loadtxt('NOSCF_ORBITALS_BETA.txt', dtype=np.complex128)

    # There are 32 occupied orbitals for each spin for the neutral system (32 alpha, 32 beta)
    mo_occ_alpha = result['mo_occ_alpha'][idx,:]
    mo_occ_beta = result['mo_occ_beta'][idx,:]
    mo_occ_alpha[32:] = np.zeros(len(mo_occ_alpha) - 32)
    mo_occ_beta[32:] = np.zeros(len(mo_occ_beta) - 32)

    neutral_mo_occ = np.zeros(len(mo_occ_alpha))
    neutral_mo_occ[:32] = np.ones(32)


    density_alpha = np.dot(alpha_coeff, np.dot(np.diag(neutral_mo_occ - mo_occ_alpha), alpha_coeff.T.conj()))
    density_beta = np.dot(beta_coeff, np.dot(np.diag(neutral_mo_occ - mo_occ_beta), beta_coeff.T.conj()))
    return density_alpha + density_beta

def make_cubes():
    mol = gto.M(verbose=0,
        atom='''
    C           0.00010        0.00000       -0.00266
    C           0.00010        0.00000        1.33266
    H           0.92879        0.00000       -0.57584
    H          -0.92884        0.00000       -0.57547
    H           0.92879        0.00000        1.90584
    H          -0.92884        0.00000        1.90547

    C           0.00010        3.00000       -0.00266
    C           0.00010        3.00000        1.33266
    H           0.92879        3.00000       -0.57584
    H          -0.92884        3.00000       -0.57547
    H           0.92879        3.00000        1.90584
    H          -0.92884        3.00000        1.90547

    C           0.00010        6.00000       -0.00266
    C           0.00010        6.00000        1.33266
    H           0.92879        6.00000       -0.57584
    H          -0.92884        6.00000       -0.57547
    H           0.92879        6.00000        1.90584
    H          -0.92884        6.00000        1.90547

    C           0.00010        9.00000       -0.00266
    C           0.00010        9.00000        1.33266
    H           0.92879        9.00000       -0.57584
    H          -0.92884        9.00000       -0.57547
    H           0.92879        9.00000        1.90584
    H          -0.92884        9.00000        1.90547
        ''',
        basis='6-31G*',
        spin = 1, charge=1)

    filename = 'ethylene_x4.out'
    result = parse_output(filename)

    den_0 = get_hole_density(result, 0) # Density at 0 au
    den_275 = get_hole_density(result, 55) # Density at 27.5 au
    den_550 = get_hole_density(result, 110) # Density at 55.0 au
    den_825 = get_hole_density(result, 165) # Density at 82.5 au

    cubegen.density(mol, 'Hole_Snapshot1_RTTDDFT.cube', den_0)
    cubegen.density(mol, 'Hole_Snapshot2_RTTDDFT.cube', den_275)
    cubegen.density(mol, 'Hole_Snapshot3_RTTDDFT.cube', den_550)
    cubegen.density(mol, 'Hole_Snapshot4_RTTDDFT.cube', den_825)

make_cubes()

def EthyleneStack():
    filename = 'ethylene_x4.out'
    result = parse_output(filename)
    time = result['time']
    hirsh_charges = result['hirsh_charge']
    mo_occ_alpha = result['mo_occ_alpha']
    mo_occ_beta = result['mo_occ_beta']
    a_charge = hirsh_charges[:,:6].sum(axis=1)
    b_charge = hirsh_charges[:,6:12].sum(axis=1)
    c_charge = hirsh_charges[:,12:18].sum(axis=1)
    d_charge = hirsh_charges[:,18:].sum(axis=1)

    plt.figure(figsize=(3.36, 2.52), dpi=600)
    ax = plt.axes()

    ax.tick_params(labelsize=8, which='both',direction='in', top=True, right=True)
    ax.yaxis.set_major_locator(MultipleLocator(0.25))
    plt.plot(time / 41.34, 16 - a_charge, 'purple', linewidth=1.5, label='A')
    plt.plot(time / 41.34, 16 - b_charge, 'blue', linewidth=1.5, label='B')
    plt.plot(time / 41.34, 16 - c_charge, 'green', linewidth=1.5, label='C')
    plt.plot(time / 41.34, 16 - d_charge, 'red', linewidth=1.5, label='D')
    plt.xlim([-0.5,10.5])
    plt.ylim([-0.01, 1.1])
    plt.ylabel('Charge', fontsize=12)
    plt.xlabel('Time (fs)', fontsize=12)
    plt.text(0.05, 0.9, 'b)', fontsize=12, va='center', transform=ax.transAxes)

    plt.savefig('ethylene_x4.png', bbox_inches='tight')


EthyleneStack()
