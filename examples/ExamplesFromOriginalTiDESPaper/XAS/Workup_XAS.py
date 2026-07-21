import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt
from matplotlib import font_manager
from matplotlib.ticker import MultipleLocator, FormatStrFormatter, MaxNLocator
import matplotlib.patches as mpatches
from tides.rt_spec import abs_spec
from tides.parse_rt import parse_output, get_length
from pyscf import lib
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

PySCFx = parse_output('x/co.out')
PySCF_dipolex = PySCFx['dipole'][:,0]
PySCFy = parse_output('y/co.out')
PySCF_dipoley = PySCFy['dipole'][:,1]
PySCFz = parse_output('z/co.out')
PySCF_time, PySCF_dipolez = PySCFz['time'][:], PySCFz['dipole'][:,2]

PySCF_dipole = np.stack((PySCF_dipolex, PySCF_dipoley, PySCF_dipolez)).T

PySCF_freq, PySCF_intensity = abs_spec(PySCF_time, PySCF_dipole, kick_str=0.01, damp=120, preprocess_zero=True)
PySCF_freq = PySCF_freq * 27.2114
PySCF_intensity = np.sum(PySCF_intensity, axis=1)

def co_xas():
    fig, axs = plt.subplots(2,1, figsize=(3.36, 2.52*2), dpi=600, sharex=False)
    plt.subplots_adjust(hspace=0.1)
    axs[0].tick_params(labelsize=8, which='both',direction='in', top=True, right=True)
    axs[0].plot(PySCF_freq, PySCF_intensity, 'r', linewidth=1.5, label='C K-Edge')
    axs[0].yaxis.set_major_locator(MultipleLocator(.8/3))
    axs[0].xaxis.set_major_locator(MultipleLocator(5))
    axs[0].set_yticklabels([])
    axs[0].set_xlim([274,291])
    axs[0].set_ylim([-0.005,1.2])
    axs[0].text(0.05, 0.9, 'a)', fontsize=12, va='center', transform=axs[0].transAxes)
    axs[0].legend(fontsize=8, loc='upper center', bbox_to_anchor=(0.7, 0.9), frameon=False, edgecolor='inherit')

    axs[1].tick_params(labelsize=8, which='both',direction='in', top=True, right=True)
    axs[1].plot(PySCF_freq, PySCF_intensity, 'r', linewidth=1.5, label='O K-Edge')
    axs[1].yaxis.set_major_locator(MultipleLocator(0.8/3))
    axs[1].xaxis.set_major_locator(MultipleLocator(5))
    axs[1].set_xlabel('Frequency (eV)', fontsize=12, labelpad=5)
    axs[1].set_yticklabels([])
    axs[1].set_xlim([514,536])
    axs[1].set_ylim([-0.005,1.2])
    axs[1].text(0.05, 0.9, 'b)', fontsize=12, va='center', transform=axs[1].transAxes)
    axs[1].legend(fontsize=8, loc='upper center', bbox_to_anchor=(0.7, 0.9), frameon=False, edgecolor='inherit')
    fig.text(-0.0, 0.5, r'Intensity (arb. units)', fontsize=12, va='center', rotation='vertical')
    plt.savefig('CO_XAS.png', bbox_inches='tight')

co_xas()
