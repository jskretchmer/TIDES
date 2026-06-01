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


def get_results(filename, dt=1.0):
    xyz = XYZReader(filename, dt=dt)
    xyz.units['time'] = 'au'

    time = []
    positions = []
    for ts in xyz:
        time.append(ts.time)
        positions.append(np.array(ts.positions).astype(np.float64))
    time = np.array(time)
    positions = np.array(positions)

    dist = get_length(positions, [1,2])
    return time, dist

time0_G_eh, _0ev_G_eh = get_results('6-31G/0eV/ehrenfest/Nn1Nf1/trajectory.xyz', dt=2.0)
time1_G_eh, _1ev_G_eh = get_results('6-31G/1eV/ehrenfest/Nn1Nf1/trajectory.xyz', dt=2.0)
time2_G_eh, _2ev_G_eh = get_results('6-31G/2eV/ehrenfest/Nn1Nf1/trajectory.xyz', dt=2.0)
time3_G_eh, _3ev_G_eh = get_results('6-31G/3eV/ehrenfest/Nn1Nf1/trajectory.xyz', dt=2.0)
time0_G_bo, _0ev_G_bo = get_results('6-31G/0eV/bomd/BOMD.md.xyz', dt=0.5)
time1_G_bo, _1ev_G_bo = get_results('6-31G/1eV/bomd/BOMD.md.xyz', dt=0.5)
time2_G_bo, _2ev_G_bo = get_results('6-31G/2eV/bomd/BOMD.md.xyz', dt=0.5)
time3_G_bo, _3ev_G_bo = get_results('6-31G/3eV/bomd/BOMD.md.xyz', dt=0.5)

time0_Gs_eh, _0ev_Gs_eh = get_results('6-31G_Star/0eV/ehrenfest/Nn1Nf1/trajectory.xyz', dt=2.0)
time1_Gs_eh, _1ev_Gs_eh = get_results('6-31G_Star/1eV/ehrenfest/Nn1Nf1/trajectory.xyz', dt=2.0)
time2_Gs_eh, _2ev_Gs_eh = get_results('6-31G_Star/2eV/ehrenfest/Nn1Nf1/trajectory.xyz', dt=2.0)
time3_Gs_eh, _3ev_Gs_eh = get_results('6-31G_Star/3eV/ehrenfest/Nn1Nf1/trajectory.xyz', dt=2.0)
time0_Gs_bo, _0ev_Gs_bo = get_results('6-31G_Star/0eV/bomd/BOMD.md.xyz', dt=0.5)
time1_Gs_bo, _1ev_Gs_bo = get_results('6-31G_Star/1eV/bomd/BOMD.md.xyz', dt=0.5)
time2_Gs_bo, _2ev_Gs_bo = get_results('6-31G_Star/2eV/bomd/BOMD.md.xyz', dt=0.5)
time3_Gs_bo, _3ev_Gs_bo = get_results('6-31G_Star/3eV/bomd/BOMD.md.xyz', dt=0.5)

def Cl2_Dissociation():

    fig, axs = plt.subplots(2,1, figsize=(3.36, 2.52*2), dpi=600, sharex=True)
    plt.subplots_adjust(hspace=0.1)
    axs[0].grid(True)
    axs[0].tick_params(labelsize=8, which='both',direction='in', top=True, right=True)
    axs[0].plot(time0_G_eh / 41.34, _0ev_G_eh, 'C0', linewidth=1.5, label='0')
    axs[0].plot(time1_G_eh / 41.34, _1ev_G_eh, 'C1', linewidth=1.5, label='1')
    axs[0].plot(time2_G_eh / 41.34, _2ev_G_eh, 'C2', linewidth=1.5, label='2')
    axs[0].plot(time3_G_eh / 41.34, _3ev_G_eh, 'C3', linewidth=1.5, label='3')
    axs[0].plot(time0_G_bo / 41.34, _0ev_G_bo, 'C0', linewidth=2.5, linestyle='--', label='0')
    axs[0].plot(time1_G_bo / 41.34, _1ev_G_bo, 'C1', linewidth=2.5, linestyle='--', label='1')
    axs[0].plot(time2_G_bo / 41.34, _2ev_G_bo, 'C2', linewidth=2.5, linestyle='--', label='2')
    axs[0].plot(time3_G_bo / 41.34, _3ev_G_bo, 'C3', linewidth=2.5, linestyle='--', label='3')

    axs[0].yaxis.set_major_locator(MultipleLocator(0.5))
    axs[0].xaxis.set_major_locator(MultipleLocator(40))

    axs[0].set_xlim([0,120])
    axs[0].set_ylim([1.5,3.99])
    axs[0].text(0.05, 0.9, 'a)', fontsize=12, va='center', transform=axs[0].transAxes)
    axs[0].text(0.03, .08, '6-31G', fontsize=15, transform=axs[0].transAxes)

    axs[1].grid(True)
    axs[1].tick_params(labelsize=8, which='both',direction='in', top=True, right=True)
    axs[1].plot(time0_Gs_eh / 41.34, _0ev_Gs_eh, 'C0', linewidth=1.5, label='0')
    axs[1].plot(time1_Gs_eh / 41.34, _1ev_Gs_eh, 'C1', linewidth=1.5, label='1')
    axs[1].plot(time2_Gs_eh / 41.34, _2ev_Gs_eh, 'C2', linewidth=1.5, label='2')
    axs[1].plot(time3_Gs_eh / 41.34, _3ev_Gs_eh, 'C3', linewidth=1.5, label='3')
    axs[1].plot(time0_Gs_bo / 41.34, _0ev_Gs_bo, 'C0', linewidth=2.5, linestyle='--', label='0')
    axs[1].plot(time1_Gs_bo / 41.34, _1ev_Gs_bo, 'C1', linewidth=2.5, linestyle='--', label='1')
    axs[1].plot(time2_Gs_bo / 41.34, _2ev_Gs_bo, 'C2', linewidth=2.5, linestyle='--', label='2')
    axs[1].plot(time3_Gs_bo / 41.34, _3ev_Gs_bo, 'C3', linewidth=2.5, linestyle='--', label='3')

    axs[1].yaxis.set_major_locator(MultipleLocator(0.5))
    axs[1].xaxis.set_major_locator(MultipleLocator(40))
    axs[1].set_xlabel('Time (fs)', fontsize=12)
    axs[1].set_xlim([0,120])
    axs[1].set_ylim([1.5,3.99])
    axs[1].text(0.05, 0.9, 'b)', fontsize=12, va='center', transform=axs[1].transAxes)
    axs[1].text(0.03, .08, r'6-31G*', fontsize=15, transform=axs[1].transAxes)
    fig.text(-0.03, 0.5, r'Distance ($\mathrm{\AA}$)', fontsize=12, va='center', rotation='vertical')

    plt.savefig('Cl2_Dissociation.png', bbox_inches='tight')

Cl2_Dissociation()


#result_G23 = parse_output('6-31G/0eV/ehrenfest/Nn2Nf3/cl2.out')
result_G32 = parse_output('6-31G/0eV/ehrenfest/Nn3Nf2/cl2.out')
result_G24 = parse_output('6-31G/0eV/ehrenfest/Nn2Nf4/cl2.out')
result_G33 = parse_output('6-31G/0eV/ehrenfest/Nn3Nf3/cl2.out')
result_G42 = parse_output('6-31G/0eV/ehrenfest/Nn4Nf2/cl2.out')
result_G11 = parse_output('6-31G/0eV/ehrenfest/Nn1Nf1/cl2.out')
#time_G23, energy_G23 = result_G23['time'], result_G23['energy']
time_G32, energy_G32 = result_G32['time'], result_G32['energy']
time_G24, energy_G24 = result_G24['time'], result_G24['energy']
time_G33, energy_G33 = result_G33['time'], result_G33['energy']
time_G42, energy_G42 = result_G42['time'], result_G42['energy']
time_G11, energy_G11 = result_G11['time'], result_G11['energy']

#result_Gs23 = parse_output('6-31G_Star/0eV/ehrenfest/Nn2Nf3/cl2.out')
result_Gs32 = parse_output('6-31G_Star/0eV/ehrenfest/Nn3Nf2/cl2.out')
result_Gs24 = parse_output('6-31G_Star/0eV/ehrenfest/Nn2Nf4/cl2.out')
result_Gs33 = parse_output('6-31G_Star/0eV/ehrenfest/Nn3Nf3/cl2.out')
result_Gs42 = parse_output('6-31G_Star/0eV/ehrenfest/Nn4Nf2/cl2.out')
result_Gs11 = parse_output('6-31G_Star/0eV/ehrenfest/Nn1Nf1/cl2.out')
#time_Gs23, energy_Gs23 = result_Gs23['time'], result_Gs23['energy']
time_Gs32, energy_Gs32 = result_Gs32['time'], result_Gs32['energy']
time_Gs24, energy_Gs24 = result_Gs24['time'], result_Gs24['energy']
time_Gs33, energy_Gs33 = result_Gs33['time'], result_Gs33['energy']
time_Gs42, energy_Gs42 = result_Gs42['time'], result_Gs42['energy']
time_Gs11, energy_Gs11 = result_Gs11['time'], result_Gs11['energy']

def Cl2_Energy_Conservation():

    fig, axs = plt.subplots(2,1, figsize=(3.36, 2.52*2), dpi=600, sharex=True)
    plt.subplots_adjust(hspace=0.1)
    axs[0].grid(True)
    axs[0].tick_params(labelsize=8, which='both',direction='in', top=True, right=True)
    #axs[0].plot(time_G23 / 41.34, (energy_G23 - energy_G23[0]) * 27.21 * 1000, linewidth=1.5, c='k', label=r'$N_n$ = 2; $N_f$ = 3')
    axs[0].plot(time_G32 / 41.34, (energy_G32 - energy_G32[0]) * 27.21 * 1000, linewidth=1.5, c='C0', label=r'$N_n$ = 3; $N_f$ = 2')
    axs[0].plot(time_G24 / 41.34, (energy_G24 - energy_G24[0]) * 27.21 * 1000, linewidth=1.5, c='C1', label=r'$N_n$ = 2; $N_f$ = 4')
    axs[0].plot(time_G33 / 41.34, (energy_G33 - energy_G33[0]) * 27.21 * 1000, linewidth=1.5, c='C2', label=r'$N_n$ = 3; $N_f$ = 3')
    axs[0].plot(time_G42 / 41.34, (energy_G42 - energy_G42[0]) * 27.21 * 1000, linewidth=1.5, c='C3', label=r'$N_n$ = 4; $N_f$ = 2')
    axs[0].plot(time_G11 / 41.34, (energy_G11 - energy_G11[0]) * 27.21 * 1000, linewidth=2.5, linestyle='--', c='k', label=r'$N_n$ = 1; $N_f$ = 1')

    axs[0].yaxis.set_major_locator(MultipleLocator(0.04))
    axs[0].xaxis.set_major_locator(MultipleLocator(40))

    axs[0].set_xlim([0,120])
    axs[0].set_ylim([-0.159,0.029])
    axs[0].text(0.05, 0.9, 'a)', fontsize=12, va='center', transform=axs[0].transAxes)
    axs[0].text(0.03, .08, '6-31G', fontsize=15, transform=axs[0].transAxes)

    axs[1].grid(True)
    axs[1].tick_params(labelsize=8, which='both',direction='in', top=True, right=True)
    #axs[1].plot(time_Gs23 / 41.34, (energy_Gs23 - energy_Gs23[0]) * 27.21 * 1000, linewidth=1.5, c='k', label=r'$N_n$ = 2; $N_f$ = 3')
    axs[1].plot(time_Gs32 / 41.34, (energy_Gs32 - energy_Gs32[0]) * 27.21 * 1000, linewidth=1.5, c='C0', label=r'$N_n$ = 3; $N_f$ = 2')
    axs[1].plot(time_Gs24 / 41.34, (energy_Gs24 - energy_Gs24[0]) * 27.21 * 1000, linewidth=1.5, c='C1', label=r'$N_n$ = 2; $N_f$ = 4')
    axs[1].plot(time_Gs33 / 41.34, (energy_Gs33 - energy_Gs33[0]) * 27.21 * 1000, linewidth=1.5, c='C2', label=r'$N_n$ = 3; $N_f$ = 3')
    axs[1].plot(time_Gs42 / 41.34, (energy_Gs42 - energy_Gs42[0]) * 27.21 * 1000, linewidth=1.5, c='C3', label=r'$N_n$ = 4; $N_f$ = 2')
    axs[1].plot(time_Gs11 / 41.34, (energy_Gs11 - energy_Gs11[0]) * 27.21 * 1000, linewidth=2.5, linestyle='--', c='k', label=r'$N_n$ = 1; $N_f$ = 1')

    axs[1].yaxis.set_major_locator(MultipleLocator(0.01))
    axs[1].xaxis.set_major_locator(MultipleLocator(40))
    axs[1].set_xlabel('Time (fs)', fontsize=12)
    axs[1].set_xlim([0,120])
    axs[1].set_ylim([-0.026,0.005])
    axs[1].text(0.05, 0.9, 'b)', fontsize=12, va='center', transform=axs[1].transAxes)
    axs[1].text(0.03, .08, '6-31G*', fontsize=15, transform=axs[1].transAxes)
    fig.text(-0.04, 0.5, r'Total Energy Conservation (meV)', fontsize=12, va='center', rotation='vertical')

    plt.savefig('Cl2_Energy_Conservation.png', bbox_inches='tight')

Cl2_Energy_Conservation()
