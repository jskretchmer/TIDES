import numpy as np

'''
SCF Time-Independent Applied Potential
'''

def static_bfield(scf, bfield):
    '''
    Since hcore is time-independent for frozen nuclei, this function calculates a modified hcore in the presence of a static B-field, and overrides the get_hcore() func to return     this modified hcore.
    '''
    x_bfield = bfield[0]
    y_bfield = bfield[1]
    z_bfield = bfield[2]

    hcore = scf.get_hcore()
    ovlp = scf.get_ovlp()
    Nsp = int(ovlp.shape[0]/2)

    ovlp = ovlp[:Nsp,:Nsp]
    hcore = hcore[:Nsp,:Nsp]

    hprime = np.zeros([2*Nsp,2*Nsp], dtype=complex)
    hprime[:Nsp,:Nsp] = hcore + 0.5 * z_bfield * ovlp
    hprime[Nsp:,Nsp:] = hcore - 0.5 * z_bfield * ovlp
    hprime[Nsp:,:Nsp] = 0.5 * (x_bfield + 1j * y_bfield) * ovlp
    hprime[:Nsp,Nsp:] = 0.5 * (x_bfield - 1j * y_bfield) * ovlp
    scf.get_hcore = lambda *args: hprime
