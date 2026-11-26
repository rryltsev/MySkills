import numpy as np
import pytest
from gfa.interface_series import stationary_thermo,authorize_sampling,core_temperature
from ase import Atoms,units

SETTINGS=dict(minimum_thermo_rows=150,mean_temperature_relative=.05,block_temperature_relative=.08,
    temperature_half_difference_relative=.03,diagonal_half_difference_GPa=.15,shear_half_difference_GPa=.10)
LIMITS=dict(Pxx=.2,Pyy=.2,Pzz=.3,Pxy=.15,Pxz=.15,Pyz=.15)

def rows(zz=0,xz=0):
    return [dict(Step=i*10,Time=i*.01,Temp=1850+10*np.sin(i*.2),Pxx=100,Pyy=-100,Pzz=zz,Pxy=50,Pxz=xz,Pyz=-50) for i in range(201)]

def test_new_trajectory_cannot_inherit_old_Cu_exception():
    gate=stationary_thermo(rows(zz=3021.4039563295455),1850,LIMITS,SETTINGS)
    assert not gate['accepted'] and not gate['checks']['pressure_means']
    with pytest.raises(RuntimeError):authorize_sampling(gate)

def test_stationary_pressure_below_new_limits_is_accepted():
    import json
    gate=stationary_thermo(rows(zz=2900,xz=-1400),1850,LIMITS,SETTINGS)
    assert gate['accepted'];authorize_sampling(gate);assert json.loads(json.dumps(gate))['accepted']

def test_zero_mean_pressure_does_not_mask_nonstationary_drift():
    trajectory=rows()
    for i,r in enumerate(trajectory):r['Pzz']=-2500+5000*i/200
    gate=stationary_thermo(trajectory,1850,LIMITS,SETTINGS)
    assert gate['checks']['pressure_means'] and not gate['checks']['pressure_stationarity']
    assert not gate['accepted']

def test_failed_check_cannot_be_bypassed_by_accepted_flag():
    with pytest.raises(RuntimeError):authorize_sampling(dict(accepted=True,checks=dict(temperature=True,front=False)))

def test_core_temperature_uses_metal_velocity_units_and_removes_drift():
    rng=np.random.default_rng(31);atoms=Atoms('Cu32Zr32',positions=np.zeros((64,3)))
    v=rng.normal(size=(64,3));m=atoms.get_masses();v-=np.average(v,axis=0,weights=m)
    target=1850.;v*=np.sqrt(3*63*units.kB*target/np.sum(m[:,None]*v*v))
    physical=v*(1000*units.fs)+np.array([100,300,-200])
    frame=dict(atoms=atoms,data={k:physical[:,i] for i,k in enumerate(['vx','vy','vz'])})
    assert core_temperature(frame,np.ones(64,bool))==pytest.approx(target,rel=1e-12)
