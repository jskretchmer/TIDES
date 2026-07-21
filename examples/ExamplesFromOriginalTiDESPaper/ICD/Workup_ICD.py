import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt
from matplotlib import font_manager
from matplotlib.ticker import MultipleLocator, FormatStrFormatter, MaxNLocator
import matplotlib.patches as mpatches
from tides.parse_rt import parse_output, get_length
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

def Water_ICD():
    filename = 'water_PD_ionize.out'

    result = parse_output(filename)

    time = result['time']
    moocc = result['mo_occ']

    fig, axs = plt.subplots(2,1, figsize=(3.36, 2.52*2), dpi=600, sharex=True)
    plt.subplots_adjust(hspace=0.1)
    axs[0].tick_params(labelsize=8, which='both',direction='in', top=True, right=True)
    axs[0].yaxis.set_major_locator(MultipleLocator(0.5))
    axs[0].plot(time / 41.34, 20 - moocc[:,:10].sum(axis=1), linewidth=1.5, c='k', label='Total')#'W1 2a1')#'PA 2a1')
    axs[0].plot(time / 41.34, 10 - moocc[:,:5].sum(axis=1), linewidth=1.5, c='b', label='Water1')#'W1 2a1')#'PA 2a1')
    axs[0].plot(time / 41.34, 10 - moocc[:,5:10].sum(axis=1), linewidth=1.5, c='r', label='Water2')#'W1 "Valence"')#'PA "Valence"')

    axs[0].set_xlim([0,45])

    axs[0].text(0.05, 0.9, 'a)', fontsize=12, va='center', transform=axs[0].transAxes)
    axs[0].legend(fontsize=8, loc='center', bbox_to_anchor=(.18, .35), frameon=False)
    axs[0].set_ylabel('Charge', fontsize=12)


    axs[1].tick_params(labelsize=8, which='both',direction='in', top=True, right=True)

    axs[1].yaxis.set_major_locator(MultipleLocator(0.5))
    axs[1].plot(time / 41.34, moocc[:,2], linewidth=1.5, c='magenta')#'W2 1b2')#'PD 1b2')

    axs[1].plot(time / 41.34, moocc[:,4], linewidth=1.5, c='cyan')#'W2 1b1')#'PD 1b1')
    axs[1].plot(time / 41.34, moocc[:,9], linewidth=1.5, c='orange')#'W1 1b1')#'PA 1b1')
    axs[1].plot(time / 41.34, moocc[:,6], linewidth=1.5, c='black')#'W1 2a1')#'PA 2a1')
    axs[1].plot(time / 41.34, moocc[:,7], linewidth=1.5, c='gray')#'W1 1b2')#'PA 1b2')


    axs[1].plot(time / 41.34, moocc[:,1], linewidth=1.5, c='blue', label=r'Water1 $\mathrm{2a_1}$')#'W2 2a1')#'PD 2a1')
    axs[1].plot(time / 41.34, moocc[:,3], linewidth=1.5, c='green', label=r'Water1 $\mathrm{3a_1}$')#'W2 3a1')#'PD 3a1')

    axs[1].plot(time / 41.34, moocc[:,8], linewidth=1.5, c='red', label=r'Water2 $\mathrm{3a_1}$')#'W1 3a1')#'PA 3a1')

    axs[1].set_xlabel('Time (fs)', fontsize=12)
    axs[1].set_ylabel('Occupation', fontsize=12)

    axs[1].set_xlim([0,45])
    axs[1].set_ylim([-.1,2.1])
    axs[1].text(0.05, 0.8, 'b)', fontsize=12, va='center', transform=axs[1].transAxes)
    axs[1].legend(fontsize=6, loc='center', bbox_to_anchor=(.5, .4), frameon=False, ncols=3, columnspacing=0.8)


    plt.savefig('Water_ICD.png', bbox_inches='tight')


Water_ICD()
