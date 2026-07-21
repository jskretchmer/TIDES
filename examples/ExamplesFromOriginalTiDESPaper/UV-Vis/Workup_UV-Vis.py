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

def benzene():
    xfilename = 'x/benzene.out'
    xresult = parse_output(xfilename)
    time = xresult['time']
    xdipole = xresult['dipole'][:,0]
    yfilename = 'y/benzene.out'
    yresult = parse_output(yfilename)
    time = yresult['time'][:]
    ydipole = yresult['dipole'][:,1]
    zfilename = 'z/benzene.out'
    zresult = parse_output(zfilename)
    time = zresult['time'][:]
    zdipole = zresult['dipole'][:,2]

    dipole = np.stack((xdipole, ydipole, zdipole)).T
    freq, intensity = abs_spec(time, dipole, kick_str=0.0001, damp=250, preprocess_zero=True)
    # Normalize so that highest visible peak has intensity of 1
    xmax = 750
    freq = freq[:xmax] * 27.2114
    intensity = np.sum(intensity[:xmax,:], axis=1)
    intensity /= intensity.max()

    plt.figure(figsize=(3.36, 2.52), dpi=600)
    ax = plt.axes()
    ax.tick_params(labelsize=8, which='both',direction='in', top=True, right=True)
    ax.xaxis.set_major_locator(MultipleLocator(5))

    plt.plot(freq, intensity, 'k', linewidth=1.5, label='RT')
    plt.xlabel('Frequency (eV)', fontsize=12)
    plt.ylabel('Intensity (arb. units)', fontsize=12, labelpad=-5)
    plt.yticks(c='w')
    plt.xlim([4,21])
    plt.ylim([-0.001,1.0])
    plt.legend(fontsize=8, loc='upper center', bbox_to_anchor=(0.375, .95), fancybox=False, framealpha=1., edgecolor='inherit')
    plt.savefig('Benzene_UV-Vis.png', bbox_inches='tight')

benzene()
