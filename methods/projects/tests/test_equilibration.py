import numpy as np
from gfa.equilibration import block_summary,motion

def test_multi_origin_brownian_diffusion_and_com():
    rng=np.random.default_rng(573)
    dt=.04;D=.3;n=400;N=80
    x=np.cumsum(rng.normal(0,np.sqrt(2*D*dt),(n,N,3)),axis=0)
    species=['Cu']*40+['Zr']*40
    base=motion(x,np.ones(N),species,dt,2.7)
    shifted=motion(x+np.arange(n)[:,None,None]*np.array([.1,-.2,.3]),np.ones(N),species,dt,2.7)
    for s in base:
        assert abs(base[s]['D_2to5ps_A2_ps']/D-1)<.2
        assert np.allclose(base[s]['self_isf'],shifted[s]['self_isf'])
        assert base[s]['tau_1e_ps'] is not None

def test_block_summary_exposes_drift():
    stats=block_summary(np.linspace(0,1,1000),100)
    assert stats['half_difference']>.49
    assert stats['block_count']==10

def test_static_solid_does_not_relax():
    rng=np.random.default_rng(23)
    x=np.tile(rng.normal(size=(30,3)),(301,1,1))
    row=motion(x,np.ones(30),['Cu']*30,.04,2.7)['Cu']
    assert row['tau_1e_ps'] is None
    assert row['Fs_at_5ps']==1
    assert row['msd_at_5ps_A2']==0
