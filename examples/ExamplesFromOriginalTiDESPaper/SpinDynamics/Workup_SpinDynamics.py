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


def lithium_trimer():
    filename = 'li_trimer.out'
    result = parse_output(filename)
    time = result['time']
    atom_mag = result['hirsh_mag']

    fig, axs = plt.subplots(2,1, figsize=(3.36, 2.52*2), dpi=600, sharex=True)
    plt.subplots_adjust(hspace=0.1)
    axs[0].tick_params(labelsize=8, which='both',direction='in', top=True, right=True)
    axs[0].plot(time / 41340, atom_mag[:,0,0], color=[1,0,0], linewidth=1.5)
    axs[0].plot(time / 41340, atom_mag[:,1,0], color=[0,0.75,0], linewidth=1.5)
    axs[0].plot(time / 41340, atom_mag[:,2,0], color=[0,0.5,0.75], linewidth=1.5)
    axs[0].yaxis.set_major_locator(MultipleLocator(0.1))

    axs[0].text(0.05, 0.75, 'a)', fontsize=12, va='center', transform=axs[0].transAxes)
    axs[0].set_ylabel(r'$\mathrm{M_x}$', fontsize=12)
    axs[0].set_xlim([0,2])
    axs[0].set_ylim([-0.25,0.25])

    axs[1].tick_params(labelsize=8, which='both',direction='in', top=True, right=True)
    axs[1].plot(time / 41340, atom_mag[:,0,2], color=[1,0,0], linewidth=1.5)
    axs[1].plot(time / 41340, atom_mag[:,1,2], color=[0,0.75,0], linewidth=1.5)
    axs[1].plot(time / 41340, atom_mag[:,2,2], color=[0,0.5,0.75], linewidth=1.5)
    axs[1].yaxis.set_major_locator(MultipleLocator(0.1))

    axs[1].set_xlabel('Time (ps)', fontsize=12)
    axs[1].set_ylabel(r'$\mathrm{M_z}$', fontsize=12)

    axs[1].text(0.05, 0.75, 'b)', fontsize=12, va='center', transform=axs[1].transAxes)
    axs[1].set_xlim([0,2])
    axs[1].set_ylim([-0.25,0.25])


    plt.savefig('LiTrimer_Mags.png', bbox_inches='tight')

    times = np.arange(0, 72000, 8268)

    m1 = atom_mag[:,:,0]
    m2 = atom_mag[:,:,2]
    plt.figure(figsize=(3.36, 2.52), dpi=600)
    ax = plt.axes()
    ax.tick_params(labelsize=8, which='both',direction='in', top=True, right=True)
    ax.xaxis.set_major_locator(MultipleLocator(1.))
    ax.yaxis.set_major_locator(MultipleLocator(1.))
    for idx, t in enumerate(times):
        plt.quiver(0.0, 0.909326674, m1[t*2,0], m2[t*2,0], color=[1-idx*0.075,0,0], scale=np.linalg.norm(atom_mag[t,0,:])+1.5, headlength=5)
        plt.quiver(-1.05, -0.909326674, m1[t*2,1], m2[t*2,1], color=[0,1-idx*0.075,0], scale=np.linalg.norm(atom_mag[t,0,:])+1.5, headlength=5)
        plt.quiver(1.05, -0.909326674, m1[t*2,2], m2[t*2,2], color=[0,0.75-idx*0.075,1-idx*0.075], scale=np.linalg.norm(atom_mag[t,0,:])+1.5, headlength=5)

    arr_image = plt.imread('QuiverLabel.png', format='png')


    axin = ax.inset_axes([0.805,0.545,0.15,0.35],transform=ax.transAxes)    # create new inset axes in data coordinates
    axin.tick_params(left=False, bottom=False, top=False, right=False)
    axin.set_xticks([])
    axin.set_yticks([])
    axin.spines['top'].set_visible(False)
    axin.spines['bottom'].set_visible(False)
    axin.spines['left'].set_visible(False)
    axin.spines['right'].set_visible(False)
    axin.imshow(arr_image)

    plt.xlim([-1.5,1.5])
    plt.ylim([-1.5,1.5])
    plt.text(.92, 0.0, '0.0 ps', fontsize=8, va='center')
    plt.text(.92, 1.3, '1.8 ps', fontsize=8, va='center')

    plt.xlabel(r'x ($\mathrm{\AA}$)', fontsize=12)
    plt.ylabel(r'z ($\mathrm{\AA}$)', fontsize=12)
    plt.savefig('LiTrimer_Quiver.png', bbox_inches='tight')

lithium_trimer()
