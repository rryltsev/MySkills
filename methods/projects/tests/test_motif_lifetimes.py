import numpy as np
from gfa.motif_lifetimes import lifetime,intermittent,describe,welch_difference
from gfa.structural_atlas import temporal_diagnostics

def test_vectorized_episodes_match_original_km_at_both_cadences():
    rng=np.random.default_rng(42);m=rng.random((90,100))<.2;species=np.arange(100)%2
    for dt in [.04,.2]:
        old=temporal_diagnostics(m,species,dt=dt);new=lifetime(m,dt)
        assert new['episode_count']==old['episode_count']
        assert new['right_censored']==old['right_censored']
        np.testing.assert_allclose(new['RMST_2ps'],old['KM_RMST_2ps'])
        corr=intermittent(m,species,dt,[dt,2*dt,5*dt,10*dt])
        for time,value in corr.items():np.testing.assert_allclose(value,old['intermittent_correlation'][str(round(float(time),8))])

def test_known_censored_runs_and_unsupported_tail_bounds():
    m=np.array([[0,1],[1,1],[1,0],[0,0],[0,1]],bool)
    r=lifetime(m,.04)
    assert r['left_censored_excluded']==1 and r['right_censored']==1
    assert r['episode_count']==2
    # A zero-duration right-censor does not cause a failure; remaining event is at .08 ps.
    assert np.isclose(r['RMST_2ps'],.08)
    empty=lifetime(np.zeros((10,2),bool),.04)
    assert empty['RMST_2ps'] is None and empty['RMST_bounds_2ps']==[0,2]

def test_species_balancing_does_not_weight_by_episode_counts():
    m=np.zeros((20,3),bool);m[1:3,:2]=True;m[1:9,2]=True
    r=describe(m,np.array([0,0,1]),.04,[.04,.2])
    np.testing.assert_allclose(r['balanced']['RMST_bounds_2ps'],[.2,.2])
    assert r['all']['RMST_2ps']<.2

def test_replica_interval_keeps_censoring_bounds_and_detects_reversal():
    cu=np.array([[.20,.21],[.21,.22],[.22,.23]]);ni=np.array([[.10,.10],[.11,.11],[.12,.12]])
    d=welch_difference(cu,ni)
    np.testing.assert_allclose(d['Cu_minus_Ni_bounds_ps'],[.10,.11])
    assert d['Welch_95_interval_ps'][0]>0
    reverse=welch_difference(ni,cu)
    assert reverse['Welch_95_interval_ps'][1]<0
