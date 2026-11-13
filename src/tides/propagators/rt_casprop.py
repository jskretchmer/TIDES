import numpy as np
from tides.observables import rt_observables
from tides.propagators import rt_integrators
from tides.utils import applyham_pyscf
from tides.utils.rt_utils import update_chkfile, print_info

# Real-Time Propagation for CAS/RAS CI/SCF

def propagate(rt_cr,mo_coeff_print):
    print(np.real(np.sum(np.diag(rt_cr.den_ao@rt_cr.ovlp)))) # Check that initial number of electrons is expected
    print_info(rt_cr,mo_coeff_print) # I don't mess with this
    rt_observables._check_observables(rt_cr) # I don't mess with this

    rt_cr._integrate_function = rt_integrators.get_integrator(rt_cr) # set integrator

    # Set Hamiltonian at current time
    rt_cr._h1e_mo = rt_cr.get_h1e_mo(rt_cr._h1e_AO_0)
    rt_cr._h2e_mo = rt_cr.get_h2e_mo()
    if rt_cr._castype == 'CASSCF':
        rt_cr._h1e_orth = rt_cr.get_h1e_orth(rt_cr._h1e_AO_0)
        rt_cr.mo_to_orth = rt_cr.get_mo_to_orth()
        rt_cr.orth_to_mo = rt_cr.mo_to_orth.conj().T

    # Shift CI Hamiltonian by ground state energy at t=0 to improve numerical stability
    xp,xao = rt_cr.get_xMat()
    e1, h1a1, h2a1 = rt_cr.get_actH(xp)
    gs = np.zeros(np.shape(rt_cr._scf.ci))
    gs[0][0] = 1.0
    hPsi = applyham_pyscf.apply_ham_pyscf_check(gs,h1a1,h2a1,rt_cr._scf.nelecas[0],rt_cr._scf.nelecas[1],rt_cr._scf.ncas,e1)
    eShift = hPsi[0][0]
    
    file_output = open(rt_cr.outName, "wb")
    file_corrdens = open(rt_cr.corName, "wb")
    fmt_str = "%20.8e"
    
    # Integration loop
    for i in range(0, int((rt_cr.max_time - rt_cr._t0) / rt_cr.timestep)): # So calculation terminates once max_time is reached after restarts
        if np.mod(i,rt_cr.frequency) == 0:
            rt_observables.get_observables(rt_cr)
            if rt_cr.chkfile is not None:
                update_chkfile(rt_cr)

        rt_cr._integrate_function(rt_cr,eShift)

    rt_observables.get_observables(rt_cr)  # Collect observables at final time
    if rt_cr.chkfile is not None:
        update_chkfile(rt_cr)