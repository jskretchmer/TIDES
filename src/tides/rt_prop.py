import numpy as np
from tides import rt_integrators
from tides import rt_observables
from tides import rt_output
from tides.rt_utils import update_chkfile, print_info

'''
Real-time Propagation
'''

def propagate(rt_scf, mo_coeff_print):
    print_info(rt_scf, mo_coeff_print)
    rt_observables._check_observables(rt_scf)

    rt_scf._integrate_function = rt_integrators.get_integrator(rt_scf)
    rt_scf._fock_orth = rt_scf.get_fock_orth(rt_scf.den_ao)
    if rt_scf.prop == 'magnus_step':
        if not hasattr(rt_scf, 'mo_coeff_orth_old'): rt_scf.mo_coeff_orth_old = rt_scf.rotate_coeff_to_orth(rt_scf._scf.mo_coeff)
    if rt_scf.prop == 'magnus_interpol':
        if not hasattr(rt_scf, '_fock_orth_n12dt'): rt_scf._fock_orth_n12dt = np.copy(rt_scf._fock_orth)
        if not hasattr(rt_scf, 'magnus_tolerance'): rt_scf.magnus_tolerance = 1e-8
        if not hasattr(rt_scf, 'magnus_maxiter'): rt_scf.magnus_maxiter = 200

    # Start propagation
    for i in range(round(rt_scf._t0 / rt_scf.timestep), round(rt_scf.max_time / rt_scf.timestep)): # So calculation terminates once max_time is reached after restarts
        if np.mod(i, rt_scf.frequency) == 0:
            rt_observables.get_observables(rt_scf)
            if rt_scf.chkfile is not None:
                update_chkfile(rt_scf)

        rt_scf._integrate_function(rt_scf)
        if rt_scf.istype('RT_Ehrenfest') and np.mod(round(rt_scf.current_time / rt_scf.timestep - 1), rt_scf.N_step * rt_scf.Ne_step) == rt_scf.N_step * rt_scf.Ne_step -1:
            rt_scf.update_force()

    rt_observables.get_observables(rt_scf)  # Collect observables at final time
    if rt_scf.chkfile is not None:
        update_chkfile(rt_scf)

