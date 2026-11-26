"""Frozen thermal/mechanical stationarity gates for moving B2/liquid interfaces."""
import numpy as np
from ase import units
from ase.neighborlist import neighbor_list
from .interface_screening import local_order,classify,periodic_front,pressure_from_atomic_stress

PRESSURES=['Pxx','Pyy','Pzz','Pxy','Pxz','Pyz']

def stationary_thermo(rows,target,limits,settings):
    # Deduplicate run boundaries, retaining all actual measurement steps.
    rows=list({int(r['Step']):r for r in rows}.values());rows.sort(key=lambda r:r['Step'])
    assert len(rows)>=settings['minimum_thermo_rows']
    x=np.array([[r[k] for k in ['Time','Temp']+PRESSURES] for r in rows]);assert np.isfinite(x).all()
    mid=(x[0,0]+x[-1,0])/2;first=x[x[:,0]<mid];last=x[x[:,0]>=mid]
    assert len(first)>0 and len(last)>0
    mean=x[:,1:].mean(axis=0);delta=last[:,1:].mean(axis=0)-first[:,1:].mean(axis=0)
    edges=np.linspace(x[0,0],x[-1,0],5);blocks=[]
    for i in range(4):
        select=(x[:,0]>=edges[i]) & ((x[:,0]<edges[i+1]) if i<3 else (x[:,0]<=edges[i+1]))
        assert select.any();blocks.append(x[select,1:].mean(axis=0).tolist())
    checks=dict(mean_temperature=abs(mean[0]/target-1)<=settings['mean_temperature_relative'],
        block_temperatures=max(abs(b[0]/target-1) for b in blocks)<=settings['block_temperature_relative'],
        temperature_stationarity=abs(delta[0])/target<=settings['temperature_half_difference_relative'],
        pressure_means=all(abs(mean[i+1])/10000<=limits[k] for i,k in enumerate(PRESSURES)),
        pressure_stationarity=all(abs(delta[i+1])/10000<=(settings['diagonal_half_difference_GPa'] if i<3 else settings['shear_half_difference_GPa']) for i in range(6)))
    checks={k:bool(v) for k,v in checks.items()}
    return dict(accepted=all(checks.values()),checks=checks,mean_temperature_K=float(mean[0]),
        mean_pressure_GPa={k:float(mean[i+1]/10000) for i,k in enumerate(PRESSURES)},
        half_difference_pressure_GPa={k:float(delta[i+1]/10000) for i,k in enumerate(PRESSURES)},
        temperature_half_difference_K=float(delta[0]),blocks_temperature_K_and_pressure_bar=blocks,
        scope='Operational stationarity of thermal/mechanical variables; moving phase fractions and latent energy may change.')

def authorize_sampling(gate):
    if not gate.get('accepted') or not gate.get('checks') or not all(gate['checks'].values()):
        raise RuntimeError('No statistical sampling before all qualification checks pass')

def core_temperature(frame,mask):
    mass=frame['atoms'].get_masses()[mask]
    v=np.c_[frame['data']['vx'],frame['data']['vy'],frame['data']['vz']][mask]/(1000*units.fs)
    if len(mass)<2:return None
    v-=np.average(v,axis=0,weights=mass)
    return float(np.sum(mass[:,None]*v*v)/(3*(len(mass)-1)*units.kB))

def inspect_frames(frames,thermo_rows,model,p):
    rows={int(r['Step']):r for r in thermo_rows};profiles=[];minimum=float('inf');error=0.
    expected=(max(rows)-min(rows))//p['measurement_dump_every_steps']+1
    assert len(frames)==expected and frames[0]['step']==min(rows) and frames[-1]['step']==max(rows)
    for f in frames:
        a=f['atoms'];s=f['species'];assert len(a)==p['N'] and np.array_equal(np.bincount(s),[p['N']//2]*2)
        distances=neighbor_list('d',a,p['classifier']['cutoff_A']);assert len(distances)
        minimum=min(minimum,float(distances.min()))
        reciprocal=2*np.pi*np.asarray(p['replication'][:2])[:,None]*np.linalg.inv(a.cell.array).T[:2]
        values=local_order(a,s,reciprocal,p['classifier']['cutoff_A']);solid=classify(values,s,model)
        profile=periodic_front(a.positions[:,2],solid,a.cell[2,2],p['classifier']['profile_bin_width_A'],p['classifier']['profile_gaussian_sigma_bins'])
        expected=np.array([rows[f['step']][k] for k in PRESSURES]);error=max(error,float(max(abs(pressure_from_atomic_stress(f)-expected))))
        profile.update(step=f['step'],time_ps=f['step']*p['timestep_ps'],core_temperature_K=None,minimum_core_width_A=None,core_ambiguous_fraction=None)
        if profile['exactly_two_fronts']:
            L=a.cell[2,2];exclusion=2*p['classifier']['cutoff_A']
            d=(a.positions[:,2,None]-np.array(profile['crossings_A'])[None,:]+L/2)%L-L/2
            core=np.min(abs(d),axis=1)>=exclusion
            index=np.floor((a.positions[:,2]%L)/L*len(profile['counts'])).astype(int)
            phase=np.asarray(profile['solid_fraction'])[index]>=.5
            masks=[core&phase,core&~phase]
            profile['minimum_core_width_A']=max(0,min(profile['domain_widths_A'])-2*exclusion)
            thresholds=np.array([model['thresholds'][str(i)] for i in s]);ambiguous=(values[:,0]>thresholds[:,0])&~solid
            profile['core_ambiguous_fraction']=float(ambiguous[core].mean()) if core.any() else None
            if min(int(m.sum()) for m in masks)>=p['stationarity']['minimum_core_atoms']:
                profile['core_temperature_K']=[core_temperature(f,m) for m in masks]
        profiles.append(profile)
    valid=[r for r in profiles if r['exactly_two_fronts']]
    temperatures=[r['core_temperature_K'] for r in profiles if r['core_temperature_K'] is not None]
    core_temp_mean=np.mean(temperatures,axis=0).tolist() if temperatures else None
    core_temp_delta=(np.mean(temperatures[len(temperatures)//2:],axis=0)-np.mean(temperatures[:len(temperatures)//2],axis=0)).tolist() if len(temperatures)>=4 else None
    checks=dict(no_overlaps=minimum>=p['geometry_minimum_distance_A'],two_fronts=len(valid)/len(profiles)>=.9,
        bulk_core_width=bool(valid) and all(r['minimum_core_width_A']>=p['classifier']['minimum_bulk_core_width_A'] for r in valid),
        core_ambiguity=bool(valid) and all(r['core_ambiguous_fraction'] is not None and r['core_ambiguous_fraction']<=.1 for r in valid),
        core_temperatures=bool(core_temp_mean) and len(temperatures)/len(profiles)>=.9 and max(abs(v/p['temperature_K']-1) for v in core_temp_mean)<=.08,
        core_temperature_stationarity=core_temp_delta is not None and max(abs(v)/p['temperature_K'] for v in core_temp_delta)<=.05,
        atomic_pressure_sum=error<1.)
    return dict(accepted=all(checks.values()),checks=checks,minimum_distance_A=minimum,pressure_sum_error_bar=error,
        core_temperature_mean_K=core_temp_mean,core_temperature_half_difference_K=core_temp_delta,profiles=profiles)
