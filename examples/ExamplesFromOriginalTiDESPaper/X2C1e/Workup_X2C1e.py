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


def Group12_X2C():
    filename = 'Zn/zn.out'
    result = parse_output(filename)
    zn_time = result['time'][:]
    zn_dipole = result['dipole'][:,:]

    filename = 'Cd/cd.out'
    result = parse_output(filename)
    cd_time = result['time'][:]
    cd_dipole = result['dipole'][:,:]

    filename = 'Hg/hg.out'
    result = parse_output(filename)
    hg_time = result['time'][:]
    hg_dipole = result['dipole'][:,:]

    zn_freq, zn_intensity = abs_spec(zn_time, zn_dipole, kick_str=0.0001, damp=0, preprocess_zero=True)
    zn_freq = zn_freq * 27.2114
    zn_intensity = zn_intensity[:,0]
    zn_intensity /= np.max(zn_intensity[:1200])

    cd_freq, cd_intensity = abs_spec(cd_time, cd_dipole, kick_str=0.0001, damp=0, preprocess_zero=True)
    cd_freq = cd_freq * 27.2114
    cd_intensity = cd_intensity[:,0]
    cd_intensity /= np.max(cd_intensity[:1200])

    hg_freq, hg_intensity = abs_spec(hg_time, hg_dipole, kick_str=0.0001, damp=0, preprocess_zero=True)
    hg_freq = hg_freq * 27.2114
    hg_intensity = hg_intensity[:,0]
    hg_intensity /= np.max(hg_intensity[:1200])

    fig, axs = plt.subplots(3,1, figsize=(3.36, 2.52*3), dpi=600, sharex=True)
    plt.subplots_adjust(hspace=0.1)
    axs[0].tick_params(labelsize=8, length=2, which='both',direction='in', top=True, right=True)
    axs[0].plot(zn_freq, zn_intensity, 'b', linewidth=1.5, label='Zn')
    axs[0].yaxis.set_major_locator(MultipleLocator(1/4))
    axs[0].xaxis.set_major_locator(MultipleLocator(5))
    axs[0].set_yticklabels([])
    axs[0].set_xlim([0,20])
    axs[0].set_ylim([-0.01,1.0])
    axs[0].text(0.05, 0.9, 'a)', fontsize=12, va='center', transform=axs[0].transAxes)
    axs[0].legend(fontsize=8, loc='upper center', bbox_to_anchor=(0.75, 0.9), fancybox=False, framealpha=1., edgecolor='inherit')

    axs[1].tick_params(labelsize=8, length=2, which='both',direction='in', top=True, right=True)
    axs[1].plot(cd_freq, cd_intensity, 'g', linewidth=1.5, label='Cd')
    axs[1].yaxis.set_major_locator(MultipleLocator(1/4))
    axs[1].xaxis.set_major_locator(MultipleLocator(5))
    axs[1].set_ylabel('Intensity (arb. units)', fontsize=12, labelpad=15)
    axs[1].set_yticklabels([])
    axs[1].set_xlim([0,20])
    axs[1].set_ylim([-0.01,1.0])
    axs[1].text(0.05, 0.9, 'b)', fontsize=12, va='center', transform=axs[1].transAxes)
    axs[1].legend(fontsize=8, loc='upper center', bbox_to_anchor=(0.75, 0.9), fancybox=False, framealpha=1., edgecolor='inherit')

    axs[2].tick_params(labelsize=8, length=2, which='both',direction='in', top=True, right=True)
    axs[2].plot(hg_freq, hg_intensity, 'r', linewidth=1.5, label='Hg')
    axs[2].yaxis.set_major_locator(MultipleLocator(1/4))
    axs[2].xaxis.set_major_locator(MultipleLocator(5))
    axs[2].set_xlabel('Frequency (eV)', fontsize=12)
    axs[2].set_yticklabels([])
    axs[2].set_xlim([0,20])
    axs[2].set_ylim([-0.01,1.0])
    axs[2].text(0.05, 0.9, 'c)', fontsize=12, va='center', transform=axs[2].transAxes)
    axs[2].legend(fontsize=8, loc='upper center', bbox_to_anchor=(0.75, 0.9), fancybox=False, framealpha=1., edgecolor='inherit')
    plt.savefig('Group12_X2C.png', bbox_inches='tight')

Group12_X2C()
