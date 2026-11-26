import numpy as np
from gfa.neighbor_dynamics import multi_fs,autocorrelation,hysteresis,bond_diagnostics,first_crossing

def test_multi_origin_fs_removes_whole_cell_translation():
    t=np.arange(12)*.04;speed=np.array([[1,0,0],[-1,0,0.]])
    xyz=t[:,None,None]*speed[None,:,:];drift=t[:,None,None]*np.array([[[3.,2.,1.]]])
    fs,msd=multi_fs(xyz+drift,np.ones(2),np.array([0,1]),.04,[3.5],5)
    np.testing.assert_allclose(fs[0,0],np.sinc(3.5*t[:6]/np.pi))
    np.testing.assert_allclose(msd[0],t[:6]**2)

def test_fft_self_correlation_matches_direct_calculation():
    rng=np.random.default_rng(7);a=rng.normal(size=(25,8));a+=np.arange(25)[:,None]
    for frame in [True,False]:
        x=a-(a.mean(axis=1,keepdims=True) if frame else a.mean());v=np.mean(x*x)
        expected=[np.mean(x[:len(x)-k]*x[k:])/v for k in range(10)]
        np.testing.assert_allclose(autocorrelation(a,9,frame),expected,atol=1e-12)

def test_hysteresis_has_entry_memory_and_reentry_after_breakage():
    band=np.array([[2],[3],[2],[1],[2],[3]],np.uint8)
    np.testing.assert_array_equal(hysteresis(band,3,2)[:,0],[0,1,1,0,0,1])
    np.testing.assert_array_equal(hysteresis(band,2,2),band>=2)

def test_identity_survival_distinguishes_return_and_counts_centers():
    # Both high and low cohorts need members of each species.
    species=np.array([0,0,1,1]);pairs=np.array([[0,1],[0,2],[1,3],[2,3]])
    state=np.array([[1,1,1,1],[0,1,1,1],[1,1,1,1]],bool);masks={'m':np.array([[1,0,1,0]]*3,bool)}
    r=bond_diagnostics(state,pairs,species,[0],2,.04,masks)
    np.testing.assert_allclose(r['pair_types']['AA']['continuous'],[1,0,0])
    np.testing.assert_allclose(r['pair_types']['AA']['intermittent'],[1,0,1])
    assert np.isclose(r['pair_types']['AA']['residual_RMST_2ps'],.04)
    assert r['pair_types']['Zr-Zr']['intermittent_connected'] is None
    for c in r['cohorts']:assert np.isclose(c['continuous'][0],1)

def test_crossing_reports_resolution_and_bracket():
    r=first_crossing([0,.04,.08],[1,.2,.1])
    assert r['first_interval'] and r['bracket_ps']==[0,.04]
    assert 0<r['time_ps']<.04
