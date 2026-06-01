import types
import numpy as np


def wrap_gpu_mf(mf_gpu):
    """
    Takes a converged gpu4pyscf SCF object and returns a CPU mf whose
    get_fock and energy_tot dispatch to the GPU.  The returned object is
    a drop-in replacement for the `scf` argument to RT_SCF / RT_Ehrenfest.

    Supports real and complex density matrices; complex DMs are split into
    real and imaginary parts so the GPU kernels (which require real input)
    are called twice and the results are recombined analytically.

    Parameters
    ----------
    mf_gpu : gpu4pyscf SCF object
        Must already be converged (mf_gpu.kernel() called).

    Returns
    -------
    mf : pyscf SCF object
        CPU-side wrapper with GPU-accelerated get_fock / energy_tot.

    Raises
    ------
    ImportError
        If cupy or gpu4pyscf are not installed.
    TypeError
        If mf_gpu is not a recognised gpu4pyscf SCF object.
    RuntimeError
        If mf_gpu has not been converged before wrapping.

    Notes
    -----
    Install dependencies (adjust cuda suffix for your CUDA version)::

        pip install cupy-cuda12x
        pip install gpu4pyscf-cuda12x

    Example
    -------
    ::

        from gpu4pyscf import dft as gpu_dft
        from tides import rt_scf
        from tides.rt_gpu import wrap_gpu_mf

        mf_gpu = gpu_dft.uks.UKS(mol).density_fit()
        mf_gpu.xc = 'CAMB3LYP'
        mf_gpu.kernel()

        mf = wrap_gpu_mf(mf_gpu)

        rt_mf = rt_scf.RT_SCF(mf, timestep, max_time, prop='ep_pc', ...)
    """
    # ── dependency checks ─────────────────────────────────────────────────────
    try:
        import cupy as cp
    except ImportError:
        raise ImportError(
            "cupy is required for GPU-accelerated RT-TDDFT.\n"
            "Install with (match your CUDA version):\n"
            "  pip install cupy-cuda11x   # CUDA 11.x\n"
            "  pip install cupy-cuda12x   # CUDA 12.x\n"
            "See https://docs.cupy.dev/en/stable/install.html for details."
        )

    try:
        from gpu4pyscf import dft as _gpu_dft  # noqa: F401 — import only to confirm presence
    except ImportError:
        raise ImportError(
            "gpu4pyscf is required for GPU-accelerated RT-TDDFT.\n"
            "Install with (match your CUDA version):\n"
            "  pip install gpu4pyscf-cuda11x   # CUDA 11.x\n"
            "  pip install gpu4pyscf-cuda12x   # CUDA 12.x\n"
            "See https://github.com/pyscf/gpu4pyscf for details."
        )

    # ── input validation ──────────────────────────────────────────────────────
    if not hasattr(mf_gpu, '_numint'):
        raise TypeError(
            "mf_gpu must be a gpu4pyscf SCF object (e.g. gpu_dft.uks.UKS). "
            f"Got {type(mf_gpu)}."
        )
    if not getattr(mf_gpu, 'converged', False):
        raise RuntimeError(
            "mf_gpu must be converged before wrapping. "
            "Call mf_gpu.kernel() first."
        )

    # ── RSH / hybrid coefficients (extracted once, closed over below) ─────────
    _omega, _alpha, _hyb = mf_gpu._numint.rsh_and_hybrid_coeff(
        mf_gpu.xc, spin=mf_gpu.mol.spin)
    _is_rsh = abs(_omega) > 1e-10

    # ── build CPU mf and transfer converged SCF results ───────────────────────
    from pyscf import dft as cpu_dft

    # Mirror the gpu4pyscf object type (UKS / RKS).
    # Currently only UKS/UHF GPU workflows are tested; RKS is included for
    # completeness but the complex-DM branch below assumes spin-index 0 exists.
    _mro_names = {c.__name__ for c in type(mf_gpu).__mro__}
    if 'UKS' in _mro_names or 'UHF' in _mro_names:
        mf = cpu_dft.UKS(mf_gpu.mol).density_fit()
    elif 'RKS' in _mro_names or 'RHF' in _mro_names:
        mf = cpu_dft.RKS(mf_gpu.mol).density_fit()
    else:
        import warnings
        warnings.warn(
            f"Unrecognised gpu4pyscf type (MRO: {sorted(_mro_names)}). "
            "Defaulting to UKS CPU wrapper; check results carefully."
        )
        mf = cpu_dft.UKS(mf_gpu.mol).density_fit()

    mf.xc        = mf_gpu.xc
    mf.mo_coeff  = mf_gpu.mo_coeff.get()
    mf.mo_energy = mf_gpu.mo_energy.get()
    mf.mo_occ    = mf_gpu.mo_occ.get()
    mf.e_tot     = float(mf_gpu.e_tot)
    mf.converged = True

    # Keep references to the original GPU methods
    _gpu_get_fock   = mf_gpu.get_fock
    _gpu_energy_tot = mf_gpu.energy_tot

    # ── patched get_fock ──────────────────────────────────────────────────────
    def _get_fock(self, dm=None, h1e=None, vhf=None, cycle=-1,
                  diis=None, diis_start_cycle=None,
                  level_shift_factor=None, damp_factor=None):
        """GPU-dispatched get_fock.  Handles real and complex DMs."""
        dm_np = np.asarray(dm) if dm is not None else None

        # Real DM: pass directly to GPU
        if dm_np is None or not np.iscomplexobj(dm_np):
            dm_gpu = cp.asarray(dm_np) if dm_np is not None else None
            result = _gpu_get_fock(dm=dm_gpu)
            return result.get() if hasattr(result, 'get') else np.asarray(result)

        # Complex DM: F(P_r + i*P_i) = F(P_r) + i * [J(P_i) - K_eff(P_i)]
        # The XC term is evaluated only on the real part (standard approximation).
        dm_r_gpu = cp.asarray(dm_np.real)
        dm_i_gpu = cp.asarray(dm_np.imag)

        fock_r    = _gpu_get_fock(dm=dm_r_gpu)
        fock_r_np = fock_r.get() if hasattr(fock_r, 'get') else np.asarray(fock_r)

        vj_i, vk_i = mf_gpu.get_jk(mf_gpu.mol, dm_i_gpu, hermi=0)
        vj_tot      = vj_i[0] + vj_i[1]
        vk_eff      = vk_i * _hyb
        if _is_rsh:
            vk_lr  = mf_gpu.get_k(mf_gpu.mol, dm_i_gpu, hermi=0, omega=_omega)
            vk_eff = vk_eff + vk_lr * (_alpha - _hyb)
        veff_i    = cp.stack([vj_tot - vk_eff[0], vj_tot - vk_eff[1]])
        veff_i_np = veff_i.get() if hasattr(veff_i, 'get') else np.asarray(veff_i)

        return fock_r_np.astype(np.complex128) + 1j * veff_i_np

    # ── patched energy_tot ────────────────────────────────────────────────────
    def _energy_tot(self, dm=None, **kw):
        """GPU-dispatched energy_tot.  Handles real and complex DMs."""
        dm_np = np.asarray(dm) if dm is not None else None

        # Real DM: pass directly to GPU
        if dm_np is None or not np.iscomplexobj(dm_np):
            dm_gpu = cp.asarray(dm_np) if dm_np is not None else None
            return float(_gpu_energy_tot(dm=dm_gpu))

        # Complex DM: E(P_r + i*P_i) = E(P_r) - 0.5 * Tr[V_eff(P_i) * P_i]
        dm_r_gpu = cp.asarray(dm_np.real)
        dm_i_gpu = cp.asarray(dm_np.imag)

        e_real = float(_gpu_energy_tot(dm=dm_r_gpu))

        vj_i, vk_i = mf_gpu.get_jk(mf_gpu.mol, dm_i_gpu, hermi=0)
        vj_tot      = vj_i[0] + vj_i[1]
        vk_eff      = vk_i * _hyb
        if _is_rsh:
            vk_lr  = mf_gpu.get_k(mf_gpu.mol, dm_i_gpu, hermi=0, omega=_omega)
            vk_eff = vk_eff + vk_lr * (_alpha - _hyb)
        veff_i = cp.stack([vj_tot - vk_eff[0], vj_tot - vk_eff[1]])
        e_corr = -0.5 * float(cp.einsum('sij,sji', veff_i, dm_i_gpu).real)

        return e_real + e_corr

    mf.get_fock   = types.MethodType(_get_fock,   mf)
    mf.energy_tot = types.MethodType(_energy_tot, mf)

    return mf
