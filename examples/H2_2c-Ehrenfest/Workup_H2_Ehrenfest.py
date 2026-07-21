import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt
from matplotlib import font_manager
from matplotlib.ticker import MultipleLocator, FormatStrFormatter, MaxNLocator
import matplotlib.patches as mpatches
from tides.parse_rt import parse_output, get_length
from MDAnalysis.coordinates.XYZ import XYZReader
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


def get_results(filename, dt=0.8):
    xyz = XYZReader(filename, dt=dt)
    xyz.units['time'] = 'au'

    time = []
    positions = []
    for ts in xyz:
        time.append(ts.time)
        positions.append(np.array(ts.positions).astype(np.float64))
    time = np.array(time)
    positions = np.array(positions)

    dist = np.array(get_length(positions, [1,2]))
    return time, dist

time_RBOMD, dist_RBOMD = get_results('RB3LYP_BOMD/BOMD.md.xyz', dt=0.16)
time_UBOMD, dist_UBOMD = get_results('UB3LYP_BOMD/BOMD.md.xyz', dt=0.16)

time_REhrenfest, dist_REhrenfest = get_results('RB3LYP_Ehrenfest/trajectory.xyz', dt=0.8)
time_UEhrenfest, dist_UEhrenfest = get_results('UB3LYP_Ehrenfest/trajectory.xyz', dt=0.8)
time_2cEhrenfest, dist_2cEhrenfest = get_results('2cB3LYP_Ehrenfest/trajectory.xyz', dt=0.8)

def H2_Dissociation():

    fig, axs = plt.subplots(1,1, figsize=(3.36, 2.52), dpi=200, sharex=True)
    plt.subplots_adjust(hspace=0.1)

    axs.set_xlabel("Time (fs)")
    axs.set_ylabel("R(H-H) (Å)")

    axs.grid(True)
    axs.tick_params(labelsize=8, which='both',direction='in', top=True, right=True)

    axs.plot(time_RBOMD * 0.024188843265857 , dist_RBOMD, color='red', alpha=0.25, linewidth=1.5, label='BOMD (RB3LYP)')
    axs.plot(time_UBOMD * 0.024188843265857 , dist_UBOMD, color='green', alpha=0.25, linewidth=1.5, label='BOMD (UB3LYP)')

    axs.plot(time_REhrenfest * 0.024188843265857 , dist_REhrenfest, color='red', linestyle='--', linewidth=1.5, label='Ehrenfest (TD-RB3LYP)')
    axs.plot(time_UEhrenfest * 0.024188843265857 , dist_UEhrenfest, color='green', linewidth=1.5, label='Ehrenfest (TD-UB3LYP)')
    axs.plot(time_2cEhrenfest * 0.024188843265857, dist_2cEhrenfest, color='blue', alpha=0.25, linewidth=1.5, label='Ehrenfest (TD-2cB3LYP)')

    axs.yaxis.set_major_locator(MultipleLocator(0.5))
    axs.xaxis.set_major_locator(MultipleLocator(40))

    axs.set_xlim([0.0,100.0])
    axs.set_ylim([0.0,6.0])
    axs.set_xticks([0.0, 20.0, 40.0, 60.0, 80.0, 100.0])
    axs.set_yticks([0.0, 1.0, 2.0, 3.0, 4.0, 5.0, 6.0])

    axs.legend(loc='upper right', fontsize=4, facecolor='white', framealpha=1.)

    plt.savefig('H2_Dissociation.png', bbox_inches='tight')

H2_Dissociation()
