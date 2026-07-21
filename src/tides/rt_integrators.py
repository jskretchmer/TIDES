import numpy as np
from scipy.linalg import expm

'''
Real-time Integrator Functions
'''

def magnus_step(rt_scf):
    '''
    C'(t+dt) = U(t)C'(t-dt)
    U(t) = exp(-i*2dt*F')
    '''

    fock_orth = rt_scf._fock_orth

    # Update time, mol is updated here if rt_scf is Ehrenfest obj
    rt_scf.update_time()

    u = expm(-1j*2*rt_scf.timestep*fock_orth)

    mo_coeff_orth_new = np.matmul(u, rt_scf.mo_coeff_orth_old)

    rt_scf.mo_coeff_orth_old = rt_scf.rotate_coeff_to_orth(rt_scf._scf.mo_coeff)
    rt_scf._scf.mo_coeff = rt_scf.rotate_coeff_to_ao(mo_coeff_orth_new)
    rt_scf.den_ao = rt_scf._scf.make_rdm1(mo_occ=rt_scf.occ)
    rt_scf._fock_orth = rt_scf.get_fock_orth(rt_scf.den_ao)

def magnus_interpol(rt_scf):
    '''
    C'(t+dt) = U(t+0.5dt)C'(t)
    U(t+0.5dt) = exp(-i*dt*F')

    1. Extrapolate F'(t+0.5dt)
    2. Propagate
    3. Build new F'(t+dt), interpolate new F'(t+0.5dt)
    4. Repeat propagation and interpolation until convergence
    '''

    mo_coeff_orth = rt_scf.rotate_coeff_to_orth(rt_scf._scf.mo_coeff)
    fock_orth_p12dt = 2 * rt_scf._fock_orth - rt_scf._fock_orth_n12dt

    # Update time, mol is updated here if rt_scf is an Ehrenfest obj
    rt_scf.update_time()

    for iteration in range(rt_scf.magnus_maxiter):
        u = expm(-1j*rt_scf.timestep*fock_orth_p12dt)

        mo_coeff_orth_pdt = np.matmul(u, mo_coeff_orth)
        mo_coeff_ao_pdt = rt_scf.rotate_coeff_to_ao(mo_coeff_orth_pdt)
        den_ao_pdt = rt_scf._scf.make_rdm1(mo_coeff=mo_coeff_ao_pdt,
                                          mo_occ=rt_scf.occ)
        #rt_scf.current_time += rt_scf.timestep
        fock_orth_pdt = rt_scf.get_fock_orth(den_ao_pdt)
        #rt_scf.current_time -= rt_scf.timestep

        if (iteration > 0 and
        abs(np.linalg.norm(den_ao_pdt)
        - np.linalg.norm(den_ao_pdt_old)) < rt_scf.magnus_tolerance):

            rt_scf._scf.mo_coeff = mo_coeff_ao_pdt
            rt_scf.den_ao = den_ao_pdt
            rt_scf.fock_orth = fock_orth_pdt
            rt_scf.fock_orth_n12dt = fock_orth_p12dt
            break
        fock_orth_p12dt = 0.5 * (rt_scf._fock_orth + fock_orth_pdt)

        den_ao_pdt_old = np.copy(den_ao_pdt)
        rt_scf._scf.mo_coeff = mo_coeff_ao_pdt
        rt_scf.den_ao = den_ao_pdt

    if (abs(np.linalg.norm(den_ao_pdt) - np.linalg.norm(den_ao_pdt_old)) 
    > rt_scf.magnus_tolerance):
        rt_scf._log.error('Magnus integrator failed to converge. Increase magnus_maxiter, or decrease timestep.')
    rt_scf._log.debug1(f'Time step converged on Magnus interation: {iteration}')
    rt_scf._fock_orth = fock_orth_pdt
    rt_scf._fock_orth_n12dt = fock_orth_p12dt


def rk4(rt_scf):
    '''
    C'(t + dt) = C'(t) + (k1/6 + k2/3 + k3/3 + k4/6)
    dC' = -i * dt * (F'C')
    '''

    fock_orth = rt_scf._fock_orth
    
    # Update time, mol is updated here if rt_scf is Ehrenfest obj
    rt_scf.update_time()

    mo_coeff_orth = rt_scf.rotate_coeff_to_orth(rt_scf._scf.mo_coeff)

    # k1
    k1 = -1j * rt_scf.timestep * (np.matmul(fock_orth,mo_coeff_orth))
    mo_coeff_orth_1 = mo_coeff_orth + 1/2 * k1

    # k2
    k2 = -1j * rt_scf.timestep * (np.matmul(fock_orth,mo_coeff_orth_1))
    mo_coeff_orth_2 = mo_coeff_orth + 1/2 * k2

    # k3
    k3 = -1j * rt_scf.timestep * (np.matmul(fock_orth,mo_coeff_orth_2))
    mo_coeff_orth_3 = mo_coeff_orth + k3

    # k4
    k4 = -1j * rt_scf.timestep * (np.matmul(fock_orth,mo_coeff_orth_3))

    mo_coeff_orth_new = mo_coeff_orth + (k1/6 + k2/3 + k3/3 + k4/6)
    mo_coeff_ao_new = rt_scf.rotate_coeff_to_ao(mo_coeff_orth_new)

    rt_scf._scf.mo_coeff = mo_coeff_ao_new
    rt_scf.den_ao = rt_scf._scf.make_rdm1(mo_occ=rt_scf.occ)
    rt_scf._fock_orth = rt_scf.get_fock_orth(rt_scf.den_ao)

INTEGRATORS = {
    'magnus_step' : magnus_step,
    'magnus_interpol' : magnus_interpol,
    'rk4' : rk4,
}

def get_integrator(rt_scf):
    return INTEGRATORS[rt_scf.prop]
