import math
import numpy as np
from scipy.linalg import expm
from tides import applyham_pyscf as applyham_pyscf
from tides import fci_mod as fci_mod
import sys

'''
Real-time Integrator Functions
'''

# CFM4 constants (Blanes & Moan, Appl. Numer. Math. 56, 2006)
_CFM4_C1 = 0.5 - math.sqrt(3) / 6   # first Gauss-Legendre node
_CFM4_C2 = 0.5 + math.sqrt(3) / 6   # second Gauss-Legendre node
_CFM4_A1 = 0.25 + math.sqrt(3) / 6  # (3 + 2√3)/12
_CFM4_A2 = 0.25 - math.sqrt(3) / 6  # (3 - 2√3)/12  (slightly negative)

def _unitary_propagator(fock_orth, dt, hermitian=True):
    '''
    Compute exp(-i*dt*F). Handles both 2D (N,N) and stacked 3D (nmat,N,N) inputs.
    If hermitian=True (no CAP), uses eigh which is 2-5x faster than Pade expm.
    If hermitian=False (CAP present), falls back to scipy expm.
    '''
    if hermitian:
        eigenvalues, eigenvectors = np.linalg.eigh(fock_orth)
        phase = np.exp(-1j * dt * eigenvalues)
        return (eigenvectors * phase[..., np.newaxis, :]) @ eigenvectors.conj().swapaxes(-2, -1)
    else:
        return expm(-1j * dt * fock_orth)


def _diis_extrapolate(fock_history, resid_history):
    '''
    Pulay DIIS extrapolation for midpoint Fock convergence acceleration.
    Solves: min ||sum_i c_i * r_i||  subject to  sum_i c_i = 1
    Returns the extrapolated midpoint Fock matrix.
    '''
    nd = len(fock_history)
    B = np.zeros((nd + 1, nd + 1))
    for i in range(nd):
        for j in range(nd):
            B[i, j] = np.dot(resid_history[i], resid_history[j])
    B[nd, :nd] = -1.0
    B[:nd, nd] = -1.0
    rhs = np.zeros(nd + 1)
    rhs[nd] = -1.0
    coeffs = np.linalg.solve(B, rhs)[:nd]
    return sum(c * f for c, f in zip(coeffs, fock_history))


def magnus_step(rt_scf):
    '''
    C'(t+dt) = U(t)C'(t-dt)
    U(t) = exp(-i*2dt*F')

    Leapfrog: uses the Fock at the current time to propagate over 2dt.
    This is explicit, cheap, but not self-consistent → energy drift.
    For better energy conservation use magnus_interpol or etrs.
    '''

    fock_orth = rt_scf._fock_orth

    # Update time, mol is updated here if rt_scf is Ehrenfest obj
    rt_scf.update_time()

    hermitian = len(rt_scf._potential) == 0
    u = _unitary_propagator(fock_orth, 2*rt_scf.timestep, hermitian=hermitian)

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

    Optional DIIS acceleration: set rt_scf.magnus_diis = True
    Optional DIIS history size:  set rt_scf.magnus_diis_space (default 6)
    '''
    use_diis   = getattr(rt_scf, 'magnus_diis', False)
    n_diis     = getattr(rt_scf, 'magnus_diis_space', 6)

    mo_coeff_orth = rt_scf.rotate_coeff_to_orth(rt_scf._scf.mo_coeff)
    fock_orth_p12dt = 2 * rt_scf._fock_orth - rt_scf._fock_orth_n12dt

    # Update time, mol is updated here if rt_scf is an Ehrenfest obj
    rt_scf.update_time()

    hermitian = len(rt_scf._potential) == 0
    diis_focks, diis_resids = [], []

    for iteration in range(rt_scf.magnus_maxiter):
        u = _unitary_propagator(fock_orth_p12dt, rt_scf.timestep, hermitian=hermitian)

        mo_coeff_orth_pdt = np.matmul(u, mo_coeff_orth)
        mo_coeff_ao_pdt = rt_scf.rotate_coeff_to_ao(mo_coeff_orth_pdt)
        den_ao_pdt = rt_scf._scf.make_rdm1(mo_coeff=mo_coeff_ao_pdt,
                                          mo_occ=rt_scf.occ)
        fock_orth_pdt = rt_scf.get_fock_orth(den_ao_pdt)

        fock_new_p12dt = 0.5 * (rt_scf._fock_orth + fock_orth_pdt)

        if (iteration > 0 and
                np.linalg.norm(den_ao_pdt - den_ao_pdt_old) < rt_scf.magnus_tolerance):
            rt_scf._scf.mo_coeff = mo_coeff_ao_pdt
            rt_scf.den_ao = den_ao_pdt
            break

        if use_diis:
            resid = (fock_new_p12dt - fock_orth_p12dt).ravel()
            diis_focks.append(fock_new_p12dt)
            diis_resids.append(resid)
            if len(diis_focks) > n_diis:
                diis_focks.pop(0)
                diis_resids.pop(0)
            if len(diis_focks) >= 2:
                try:
                    fock_orth_p12dt = _diis_extrapolate(diis_focks, diis_resids)
                except np.linalg.LinAlgError:
                    fock_orth_p12dt = fock_new_p12dt
            else:
                fock_orth_p12dt = fock_new_p12dt
        else:
            fock_orth_p12dt = fock_new_p12dt

        den_ao_pdt_old = np.copy(den_ao_pdt)
        rt_scf._scf.mo_coeff = mo_coeff_ao_pdt
        rt_scf.den_ao = den_ao_pdt

    if (np.linalg.norm(den_ao_pdt - den_ao_pdt_old)
    > rt_scf.magnus_tolerance):
        rt_scf._log.error('Magnus integrator failed to converge. Increase magnus_maxiter, or decrease timestep.')
    rt_scf._log.debug1(f'Time step converged on Magnus iteration: {iteration}')
    rt_scf._fock_orth = fock_orth_pdt
    rt_scf._fock_orth_n12dt = fock_orth_p12dt


def etrs(rt_scf):
    '''
    ETRS: Enforced Time-Reversal Symmetry propagator.
    C(t+dt) = exp(-i*dt/2*F(t+dt)) @ exp(-i*dt/2*F(t)) @ C(t)

    Self-consistent: F(t+dt) depends on C(t+dt) which depends on F(t+dt).
    Algorithm:
      1. Predictor: linearly extrapolate F_pred(t+dt) = 2*F(t) - F(t-dt)
      2. U_t = exp(-i*dt/2*F(t))
      3. Iterate:
           C_pred = exp(-i*dt/2*F_pred) @ U_t @ C(t)
           Build F_new(t+dt) from C_pred
           If converged, accept
           Else F_pred = F_new, repeat
    Unitary and time-reversible by construction → excellent energy conservation.
    Cost: 2+ Fock builds/step (1 predictor build + 1 per iteration).
    See: Castro et al., J. Chem. Phys. 121, 3425 (2004).

    Uses magnus_maxiter and magnus_tolerance from rt_scf if present.
    '''
    maxiter   = getattr(rt_scf, 'magnus_maxiter', 20)
    tolerance = getattr(rt_scf, 'magnus_tolerance', 1e-7)

    fock_orth_t = rt_scf._fock_orth
    mo_coeff_orth = rt_scf.rotate_coeff_to_orth(rt_scf._scf.mo_coeff)
    hermitian = len(rt_scf._potential) == 0
    dt = rt_scf.timestep

    # Half-step propagator at t (does not change during iteration)
    U_half_t = _unitary_propagator(fock_orth_t, dt / 2, hermitian=hermitian)
    C_half = np.matmul(U_half_t, mo_coeff_orth)

    # Predictor for F(t+dt): linear extrapolation from t and t-dt
    if hasattr(rt_scf, '_etrs_fock_prev'):
        fock_pred_pdt = 2 * fock_orth_t - rt_scf._etrs_fock_prev
    else:
        fock_pred_pdt = fock_orth_t  # first step: no history, use F(t)

    rt_scf.update_time()

    den_ao_pdt_old = None
    for iteration in range(maxiter):
        U_half_pdt = _unitary_propagator(fock_pred_pdt, dt / 2, hermitian=hermitian)
        mo_coeff_orth_pdt = np.matmul(U_half_pdt, C_half)
        mo_coeff_ao_pdt = rt_scf.rotate_coeff_to_ao(mo_coeff_orth_pdt)
        den_ao_pdt = rt_scf._scf.make_rdm1(mo_coeff=mo_coeff_ao_pdt,
                                            mo_occ=rt_scf.occ)
        fock_orth_pdt = rt_scf.get_fock_orth(den_ao_pdt)

        if (den_ao_pdt_old is not None and
                np.linalg.norm(den_ao_pdt - den_ao_pdt_old) < tolerance):
            break

        fock_pred_pdt = fock_orth_pdt
        den_ao_pdt_old = np.copy(den_ao_pdt)
        rt_scf._scf.mo_coeff = mo_coeff_ao_pdt
        rt_scf.den_ao = den_ao_pdt

    if (den_ao_pdt_old is not None and
            np.linalg.norm(den_ao_pdt - den_ao_pdt_old) > tolerance):
        rt_scf._log.error('ETRS integrator failed to converge. Increase magnus_maxiter, or decrease timestep.')
    rt_scf._log.debug1(f'ETRS converged on iteration: {iteration}')

    rt_scf._etrs_fock_prev = fock_orth_t
    rt_scf._scf.mo_coeff = mo_coeff_ao_pdt
    rt_scf.den_ao = den_ao_pdt
    rt_scf._fock_orth = fock_orth_pdt


def rk4(rt_scf):
    '''
    C'(t + dt) = C'(t) + (k1/6 + k2/3 + k3/3 + k4/6)
    dC' = -i * dt * (F'C')
    Note: uses F at the start of the step throughout (no midpoint Fock update).

    Non-unitary: density norm drifts over time. QR re-orthogonalization is applied
    after each step to limit norm drift and prevent numerical blowup, but energy
    conservation is still poor compared to unitary integrators.
    '''

    fock_orth = rt_scf._fock_orth

    # Update time, mol is updated here if rt_scf is Ehrenfest obj
    rt_scf.update_time()

    mo_coeff_orth = rt_scf.rotate_coeff_to_orth(rt_scf._scf.mo_coeff)

    # k1
    k1 = -1j * rt_scf.timestep * (np.matmul(fock_orth, mo_coeff_orth))
    mo_coeff_orth_1 = mo_coeff_orth + 1/2 * k1

    # k2
    k2 = -1j * rt_scf.timestep * (np.matmul(fock_orth, mo_coeff_orth_1))
    mo_coeff_orth_2 = mo_coeff_orth + 1/2 * k2

    # k3
    k3 = -1j * rt_scf.timestep * (np.matmul(fock_orth, mo_coeff_orth_2))
    mo_coeff_orth_3 = mo_coeff_orth + k3

    # k4
    k4 = -1j * rt_scf.timestep * (np.matmul(fock_orth, mo_coeff_orth_3))

    mo_coeff_orth_new = mo_coeff_orth + (k1/6 + k2/3 + k3/3 + k4/6)

    # QR re-orthogonalization: restores column orthonormality lost due to non-unitarity.
    # Prevents norm blowup but does not make RK4 energy-conserving.
    Q, _ = np.linalg.qr(mo_coeff_orth_new)
    mo_coeff_orth_new = Q

    mo_coeff_ao_new = rt_scf.rotate_coeff_to_ao(mo_coeff_orth_new)

    rt_scf._scf.mo_coeff = mo_coeff_ao_new
    rt_scf.den_ao = rt_scf._scf.make_rdm1(mo_occ=rt_scf.occ)
    rt_scf._fock_orth = rt_scf.get_fock_orth(rt_scf.den_ao)

def rk4cr(rt_cr,fo,fs,fc,eShift):
    '''
    i d/dt|r> = sum(s) X(sr)|s>
    i d/dt C(I) = sum(J) H(JI)C(J)-X(JI)C(J)
    '''
    # Note function f in comments represents derivative equation

    # Call to update MO coefficient at new time step
    def updateMO(moNew):
        rt_cr.mo_to_ao = np.copy(moNew)
        rt_cr.ao_to_mo = rt_cr.get_ao_to_mo()
        rt_cr._scf.mo_coeff[:,:rt_cr.numP] = np.copy(rt_cr.mo_to_ao)
        rt_cr._h1e_mo = rt_cr.get_h1e_mo()
        rt_cr._h2e_mo = rt_cr.get_h2e_mo()
        rt_cr.mo_to_orth = rt_cr.get_mo_to_orth()
        rt_cr.orth_to_mo = rt_cr.mo_to_orth.conj().T

    # Call to update Hamiltonian at new time step
    def updateHam():
        rt_cr._h1e_orth = rt_cr.get_h1e_orth()
        rt_cr._h1e_mo = rt_cr.get_h1e_mo()
        rt_cr._h2e_orth = rt_cr.get_h2e_orth()
        rt_cr._h2e_mo = rt_cr.get_h2e_mo()

    # Collect initial terms
    x_mat = rt_cr.get_x()
    e0, h1Act, h2Act = rt_cr.get_embH(x_mat)
    reci0 = np.copy(rt_cr._scf.ci.real)
    imci0 = np.copy(rt_cr._scf.ci.imag)
    c0 = np.copy(rt_cr._scf.ci)
    mo0 = np.copy(rt_cr.mo_to_ao)
    
    # k1 = f(t0,y0)
    ck1 = (-1j*applyham_pyscf.apply_ham_pyscf_check(reci0,h1Act,h2Act,rt_cr._scf.nelecas[0],rt_cr._scf.nelecas[1],rt_cr._scf.ncas,e0-eShift))+(applyham_pyscf.apply_ham_pyscf_check(imci0,h1Act,h2Act,rt_cr._scf.nelecas[0],rt_cr._scf.nelecas[1],rt_cr._scf.ncas,e0-eShift))
    rk1 = -1j*np.matmul(rt_cr.mo_to_ao,x_mat)

    # c1 and mo1 represent y0 + k1*timestep/2
    c1 = rt_cr._scf.ci + (rt_cr.timestep*ck1/2)
    mo1 = rt_cr.mo_to_ao +(rt_cr.timestep*rk1/2)

    # Update system
    if rt_cr._castype == 'CASSCF':
        updateMO(mo1)
    rt_cr.update_time()
    rt_cr._scf.ci = np.copy(c1)
    rt_cr.den_ao = rt_cr.get_den_ao()
    if len(rt_cr._potential) > 0:
        updateHam()
    x2 = rt_cr.get_x()
    e2, h1a2, h2a2 = rt_cr.get_embH(x2)
    reci1 = np.copy(c1.real)
    imci1 = np.copy(c1.imag)

    # k2 = f(t0 + timestep/2,y0 + k1*timestep/2)
    ck2 = (-1j*applyham_pyscf.apply_ham_pyscf_check(reci1,h1a2,h2a2,rt_cr._scf.nelecas[0],rt_cr._scf.nelecas[1],rt_cr._scf.ncas,e2-eShift))+(applyham_pyscf.apply_ham_pyscf_check(imci1,h1a2,h2a2,rt_cr._scf.nelecas[0],rt_cr._scf.nelecas[1],rt_cr._scf.ncas,e2-eShift))
    rk2 = -1j*np.matmul(rt_cr.mo_to_ao,x2)

    # c2 and mo2 represent y0 + k2*timestep/2
    c2 = c0 + (rt_cr.timestep*ck2/2)
    mo2 = mo0 + (rt_cr.timestep*rk2/2)

    # Update system. Note time didn't increment but ci coefficients and molecular orbitals are updated
    if rt_cr._castype == 'CASSCF':
        updateMO(mo2)
    rt_cr._scf.ci = np.copy(c2)
    rt_cr.den_ao = rt_cr.get_den_ao()
    x3 = rt_cr.get_x()
    e3, h1a3, h2a3 = rt_cr.get_embH(x3)
    reci2 = np.copy(c2.real)
    imci2 = np.copy(c2.imag)

    # k3 = f(t0 + timestep/2,y0 + k2*timestep/2)
    ck3 = (-1j*applyham_pyscf.apply_ham_pyscf_check(reci2,h1a3,h2a3,rt_cr._scf.nelecas[0],rt_cr._scf.nelecas[1],rt_cr._scf.ncas,e3-eShift))+(applyham_pyscf.apply_ham_pyscf_check(imci2,h1a3,h2a3,rt_cr._scf.nelecas[0],rt_cr._scf.nelecas[1],rt_cr._scf.ncas,e3-eShift))
    rk3 = -1j*np.matmul(rt_cr.mo_to_ao,x3)

    # c3 and mo3 represent y0 + k3*timestep
    c3 = c0 + (rt_cr.timestep*ck3)
    mo3 = mo0 + (rt_cr.timestep*rk3)

    # Update system
    rt_cr.update_time()
    if rt_cr._castype == 'CASSCF':
        updateMO(mo3)
    rt_cr._scf.ci = np.copy(c3)
    rt_cr.den_ao = rt_cr.get_den_ao()
    if len(rt_cr._potential) > 0:
        updateHam()
    x4 = rt_cr.get_x()
    e4, h1a4, h2a4 = rt_cr.get_embH(x4)
    reci3 = np.copy(c3.real)
    imci3 = np.copy(c3.imag)

    # k4 = f(t0 + timestep,y0 + k3*timestep)
    ck4 = (-1j*applyham_pyscf.apply_ham_pyscf_check(reci3,h1a4,h2a4,rt_cr._scf.nelecas[0],rt_cr._scf.nelecas[1],rt_cr._scf.ncas,e4-eShift))+(applyham_pyscf.apply_ham_pyscf_check(imci3,h1a4,h2a4,rt_cr._scf.nelecas[0],rt_cr._scf.nelecas[1],rt_cr._scf.ncas,e4-eShift))
    rk4 = -1j*np.matmul(rt_cr.mo_to_ao,x4)

    # y1 = (timestep/6)(k1 + 2*k2 + 2*k3 + k4)
    cf = c0 + ((rt_cr.timestep/6)*(ck1+(2*ck2)+(2*ck3)+ck4))
    mof = mo0 + ((rt_cr.timestep/6)*(rk1+(2*rk2)+(2*rk3)+rk4))

    # Update system. Note time doesn't increment
    if rt_cr._castype == 'CASSCF':
        updateMO(mof)
    rt_cr._scf.ci = np.copy(cf)
    #print(cf)
    rt_cr.den_ao = rt_cr.get_den_ao()

    # Collect output file checks
    ef, h1f, h2f = rt_cr.get_embH(np.zeros((rt_cr.numP,rt_cr.numP)))
    output = np.zeros(3)
    output[0] = rt_cr.current_time
    output[1] = fci_mod.get_FCI_E(
                h1f,
                h2f,
                ef,
                cf,
                rt_cr._scf.ncas,
                rt_cr._scf.nelecas[0],
                rt_cr._scf.nelecas[1],
                gen=False,
            )
    output[2] = np.real(np.sum(np.diag(rt_cr.den_ao@rt_cr.ovlp))) # Gives number of electrons. Shouldn't ever change.
    print(output[2])
    
    # Print MO occupation numbers for monitoring purposes
    corr1RDMmo = np.zeros((rt_cr.numP,rt_cr.numP)).astype(np.complex128)
    for a in range(rt_cr._scf.ncore):
        corr1RDMmo[a][a] = 2
    for a in range(rt_cr._scf.ncas):
        for b in range(rt_cr._scf.ncas):
            corr1RDMmo[a+rt_cr._scf.ncore][b+rt_cr._scf.ncore] = rt_cr.casrdm1[a][b]
    print(np.real(np.diag(corr1RDMmo)))

    # corrdens represents AO occupation
    diagcorr1RDM = np.real(np.diag(rt_cr.den_ao@rt_cr.ovlp))
    corrdens = np.copy(diagcorr1RDM)
    corrdens = np.insert(corrdens, 0, rt_cr.current_time)
    
    np.savetxt(fo, output.reshape(1, output.shape[0]), fs)
    fo.flush()
    np.savetxt(fc, corrdens.reshape(1, corrdens.shape[0]), fs)
    fc.flush()
    sys.stdout.flush()

def vv(rt_cr,fo,fs,fc,eShift):
    '''
    Velocity verlet integrator as shown in J. Chem. Theory Comput. 2018, 14, 8, 4129–4138
    This procedure represents equations 4-16
    For TDCASCI only
    '''

    # Update Hamiltonian whenever time increments
    def updateHam():
        rt_cr._h1e_orth = rt_cr.get_h1e_orth()
        rt_cr._h1e_mo = rt_cr.get_h1e_mo()
        rt_cr._h2e_orth = rt_cr.get_h2e_orth()
        rt_cr._h2e_mo = rt_cr.get_h2e_mo()

    # Initialize terms
    x_mat = rt_cr.get_x()
    q0 = np.copy(rt_cr._scf.ci.real)

    if rt_cr.firstStep == True:
        e1, h1a1, h2a1 = rt_cr.get_embH(x_mat)
        p0 = np.copy(rt_cr._scf.ci.imag)
        # Eq 7
        pDot0 = -applyham_pyscf.apply_ham_pyscf_check(q0,h1a1,h2a1,rt_cr._scf.nelecas[0],rt_cr._scf.nelecas[1],rt_cr._scf.ncas,e1-eShift).astype(np.float64)
        # Eq 8
        pHalfH = p0 + (rt_cr.timestep*pDot0/2)

    if rt_cr.firstStep == False:
        # Eq 12
        pHalfH = rt_cr.pMinusHalf + (rt_cr.timestep*rt_cr.pDotH)

    # Increment Time
    rt_cr.update_time()
    if len(rt_cr._potential) > 0:
        updateHam()

    e2, h1a2, h2a2 = rt_cr.get_embH(x_mat)
    # Eq 9/13
    qDotHalfH = applyham_pyscf.apply_ham_pyscf_check(pHalfH,h1a2,h2a2,rt_cr._scf.nelecas[0],rt_cr._scf.nelecas[1],rt_cr._scf.ncas,e2-eShift).astype(np.float64)
    # Eq 10/14
    qH = q0 + (rt_cr.timestep*qDotHalfH)

    # Increment Time
    rt_cr.update_time()
    if len(rt_cr._potential) > 0:
        updateHam()

    e3, h1a3, h2a3 = rt_cr.get_embH(x_mat)
    # Eq 11/15
    pDotH = -applyham_pyscf.apply_ham_pyscf_check(qH,h1a3,h2a3,rt_cr._scf.nelecas[0],rt_cr._scf.nelecas[1],rt_cr._scf.ncas,e3-eShift).astype(np.float64)
    # Eq 16
    pH = pHalfH + (rt_cr.timestep*pDotH/2)

    # Update system to new timestep
    rt_cr._scf.ci = qH+(1j*pH)
    rt_cr.pMinusHalf = np.copy(pHalfH) # Preps Eq 12 for next step
    rt_cr.pDotH = np.copy(pDotH) # Preps Eq 12 for next step
    rt_cr.firstStep = False
    rt_cr.den_ao = rt_cr.get_den_ao()

    # Collect output file checks
    ef, h1f, h2f = rt_cr.get_embH(np.zeros((rt_cr.numP,rt_cr.numP)))
    output = np.zeros(3)
    output[0] = rt_cr.current_time
    output[1] = fci_mod.get_FCI_E(
                h1f,
                h2f,
                ef,
                rt_cr._scf.ci,
                rt_cr._scf.ncas,
                rt_cr._scf.nelecas[0],
                rt_cr._scf.nelecas[1],
                gen=False,
            )
    diagcorr1RDM = np.real(np.diag(rt_cr.den_ao@rt_cr.ovlp))
    # corrdens stores AO occupations at the given time step
    corrdens = np.copy(diagcorr1RDM)
    output[2] = np.real(np.sum(np.diag(rt_cr.den_ao@rt_cr.ovlp))) # Gives number of electrons. Shouldn't ever change.
    print(output[2])
    corrdens = np.insert(corrdens, 0, rt_cr.current_time)
    
    np.savetxt(fo, output.reshape(1, output.shape[0]), fs)
    fo.flush()
    np.savetxt(fc, corrdens.reshape(1, corrdens.shape[0]), fs)
    fc.flush()
    sys.stdout.flush()
    


def ep_pc(rt_scf):
    '''
    EP-PC: Exponential density Predictor/Corrector (EP-PC1 variant).
    Zhu & Herbert, J. Chem. Phys. 148, 044117 (2018), Algorithm 2.

    Each time step:
      Step 2 — Predictor: full MMUT step using F_N (no Fock build)
                 P^p = exp(-iΔt·F_N) P_N exp(+iΔt·F_N)
      Step 3 — Build F^p from P^p   [1 Fock build]
      Step 4 — Corrector: trapezoidal average propagator
                 U = exp(-iΔt/2·(F_N + F^p)),  P^c = U·P_N·U†
      Step 5 — Check ||P^p - P^c||_F < tolerance
               If not converged: P^p ← P^c, go to Step 3

    The MMUT predictor gives a far better starting density than linear Fock
    extrapolation, so the corrector typically converges in 1 iteration
    → ~2 Fock builds/step at Δt=0.5 a.u. (vs ~8 for magnus_interpol).

    Uses magnus_maxiter and magnus_tolerance from rt_scf if present.
    '''
    maxiter   = getattr(rt_scf, 'magnus_maxiter', 20)
    tolerance = getattr(rt_scf, 'magnus_tolerance', 1e-7)

    fock_orth_N     = rt_scf._fock_orth
    mo_coeff_orth_N = rt_scf.rotate_coeff_to_orth(rt_scf._scf.mo_coeff)
    hermitian       = len(rt_scf._potential) == 0
    dt              = rt_scf.timestep

    rt_scf.update_time()

    # Step 2: predictor — full MMUT step, no Fock build required
    U_full           = _unitary_propagator(fock_orth_N, dt, hermitian=hermitian)
    mo_coeff_ao_pred = rt_scf.rotate_coeff_to_ao(np.matmul(U_full, mo_coeff_orth_N))
    den_ao_pred      = rt_scf._scf.make_rdm1(mo_coeff=mo_coeff_ao_pred, mo_occ=rt_scf.occ)

    fock_orth_pdt   = None
    mo_coeff_ao_pdt = None
    den_ao_pdt      = None
    converged       = False

    for iteration in range(maxiter):
        # Step 3: build F^p from current predicted density
        fock_orth_pdt = rt_scf.get_fock_orth(den_ao_pred)

        # Step 4: corrector — single exp of trapezoidal-averaged Fock
        F_avg            = 0.5 * (fock_orth_N + fock_orth_pdt)
        U_avg            = _unitary_propagator(F_avg, dt, hermitian=hermitian)
        mo_coeff_ao_pdt  = rt_scf.rotate_coeff_to_ao(np.matmul(U_avg, mo_coeff_orth_N))
        den_ao_pdt       = rt_scf._scf.make_rdm1(mo_coeff=mo_coeff_ao_pdt, mo_occ=rt_scf.occ)

        # Step 5: consistency check — Eq. (21) of Zhu & Herbert: ||ΔP||_F / (n·α) < ξ
        # n = basis dimension (last axis; den_ao is (n,n) for RHF or (2,n,n) for UKS)
        # α = largest eigenvalue of P_pred
        n_basis = den_ao_pred.shape[-1]
        alpha = float(np.linalg.eigvalsh(den_ao_pred).max())
        if np.linalg.norm(den_ao_pred - den_ao_pdt) / (n_basis * alpha) < tolerance:
            converged = True
            break

        # EP-PC1: update predictor from corrector and repeat
        den_ao_pred      = den_ao_pdt
        rt_scf._scf.mo_coeff = mo_coeff_ao_pdt
        rt_scf.den_ao    = den_ao_pdt

    if not converged:
        rt_scf._log.error('EP-PC integrator failed to converge. Increase magnus_maxiter, or decrease timestep.')
    rt_scf._log.debug1(f'EP-PC converged on iteration: {iteration}')

    rt_scf._scf.mo_coeff = mo_coeff_ao_pdt
    rt_scf.den_ao        = den_ao_pdt
    rt_scf._fock_orth    = fock_orth_pdt   # F_{N+1} consistent with P_{N+1}


def cfm4(rt_scf):
    '''
    CFM4: Commutator-Free Magnus 4th-order integrator (self-consistent variant).

    φ(t+dt) = exp(-iΔt(α₁F₁ + α₂F₂)) exp(-iΔt(α₂F₁ + α₁F₂)) φ(t)

    F₁ = F(t + c₁·dt),  F₂ = F(t + c₂·dt)  (Gauss-Legendre quadrature nodes)
      c₁ = 1/2 - √3/6 ≈ 0.211,  c₂ = 1/2 + √3/6 ≈ 0.789
      α₁ = 1/4 + √3/6 ≈ 0.539,  α₂ = 1/4 - √3/6 ≈ -0.039

    F₁ and F₂ are obtained by LINEAR INTERPOLATION between F(t) and F(t+dt):
      F₁ = (1 - c₁)·F(t) + c₁·F(t+dt)
      F₂ = (1 - c₂)·F(t) + c₂·F(t+dt)

    F(t+dt) is obtained self-consistently:
      Predictor: F_pred(t+dt) = 2·F(t) - F(t-dt)  [linear extrapolation]
      Corrector: propagate with current F₁,F₂, build F_new(t+dt), repeat until convergence.

    This avoids the catastrophic amplification of the forward Lagrange extrapolation
    approach (which uses weights up to ±4 for large dt) and gives stable energy conservation.

    Cost: 2+ Fock builds/step.  Order: 4 (with converged self-consistency).
    See: Gómez Pueyo et al., JCTC 2018, eq 54;
         Blanes & Moan, Appl. Numer. Math. 56, 1519 (2006), eq 43.
    '''
    maxiter   = getattr(rt_scf, 'magnus_maxiter', 20)
    tolerance = getattr(rt_scf, 'magnus_tolerance', 1e-7)

    fock_orth_t = rt_scf._fock_orth
    mo_coeff_orth = rt_scf.rotate_coeff_to_orth(rt_scf._scf.mo_coeff)
    hermitian = len(rt_scf._potential) == 0
    dt = rt_scf.timestep

    # Predictor for F(t+dt)
    if hasattr(rt_scf, '_cfm4_fock_prev'):
        fock_pred_pdt = 2 * fock_orth_t - rt_scf._cfm4_fock_prev
    else:
        fock_pred_pdt = fock_orth_t  # first step: no history

    rt_scf.update_time()

    den_ao_pdt_old = None
    for iteration in range(maxiter):
        # Linear interpolation to quadrature nodes
        F_t1 = (1 - _CFM4_C1) * fock_orth_t + _CFM4_C1 * fock_pred_pdt
        F_t2 = (1 - _CFM4_C2) * fock_orth_t + _CFM4_C2 * fock_pred_pdt

        H_A = _CFM4_A1 * F_t1 + _CFM4_A2 * F_t2
        H_B = _CFM4_A2 * F_t1 + _CFM4_A1 * F_t2

        # Apply right-to-left: U_A @ U_B @ C(t)
        U_B = _unitary_propagator(H_B, dt, hermitian=hermitian)
        U_A = _unitary_propagator(H_A, dt, hermitian=hermitian)
        mo_coeff_orth_pdt = np.matmul(U_A, np.matmul(U_B, mo_coeff_orth))
        mo_coeff_ao_pdt = rt_scf.rotate_coeff_to_ao(mo_coeff_orth_pdt)
        den_ao_pdt = rt_scf._scf.make_rdm1(mo_coeff=mo_coeff_ao_pdt,
                                            mo_occ=rt_scf.occ)
        fock_orth_pdt = rt_scf.get_fock_orth(den_ao_pdt)

        if (den_ao_pdt_old is not None and
                np.linalg.norm(den_ao_pdt - den_ao_pdt_old) < tolerance):
            break

        fock_pred_pdt = fock_orth_pdt
        den_ao_pdt_old = np.copy(den_ao_pdt)
        rt_scf._scf.mo_coeff = mo_coeff_ao_pdt
        rt_scf.den_ao = den_ao_pdt

    if (den_ao_pdt_old is not None and
            np.linalg.norm(den_ao_pdt - den_ao_pdt_old) > tolerance):
        rt_scf._log.error('CFM4 integrator failed to converge. Increase magnus_maxiter, or decrease timestep.')
    rt_scf._log.debug1(f'CFM4 converged on iteration: {iteration}')

    rt_scf._cfm4_fock_prev = fock_orth_t
    rt_scf._scf.mo_coeff = mo_coeff_ao_pdt
    rt_scf.den_ao = den_ao_pdt
    rt_scf._fock_orth = fock_orth_pdt


INTEGRATORS = {
    'magnus_step'    : magnus_step,
    'magnus_interpol': magnus_interpol,
    'etrs'           : etrs,
    'ep_pc'          : ep_pc,
    'rk4'            : rk4,
    'cfm4'           : cfm4,
    'rk4cr': rk4cr,
    'vv': vv
}

def get_integrator(rt_scf):
    return INTEGRATORS[rt_scf.prop]
