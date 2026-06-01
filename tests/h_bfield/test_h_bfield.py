import pytest
from pathlib import Path
from pyscf import gto, scf, dft
from tides import rt_scf, rt_vapp, parse_rt
from tides.staticfield import static_bfield

dir_path = str(Path(__file__).resolve().parent)

def test_h_bfield():
    # same calculation as TIDES/examples/H_BField
    mag_z = 0.000085
    mol = gto.M(
    	verbose = 0,
    	atom='H 0 0 0',
    	basis='STO-3G',
        spin = 1)
    mf = scf.ghf.GHF(mol)
    mf.kernel()
    static_bfield(mf, [0,0,mag_z])
    rt_mf = rt_scf.RT_SCF(mf, 1.0, 25, filename = dir_path + '/output.out')
    rt_mf.prop = 'magnus_step' 
    rt_mf.observables.update(mag=True)
    rt_mf.kernel()
    # parse data for comparison
    data = parse_rt.parse_output(dir_path + '/output.out')
    data_ref = parse_rt.parse_output(dir_path + '/output.ref')
    # confirm final magy equal to 8 digits
    assert round(data['mag'][-1,1], 8) == round(data_ref['mag'][-1,1], 8)

