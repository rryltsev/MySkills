"""Discrete-cadence motif episodes, censored survival and replica comparisons."""
import numpy as np
from scipy.stats import t as student_t

def lifetime(masks,dt,tau=2.):
    """KM on observed run lengths, with no extrapolation of unsupported tails."""
    masks=np.asarray(masks,bool);assert masks.ndim==2 and len(masks)>=2 and dt>0
    frames,n=masks.shape
    changes=np.diff(np.pad(masks.T,((0,0),(1,1))).astype(np.int8),axis=1)
    atom,start=np.where(changes==1);atom_end,end=np.where(changes==-1)
    np.testing.assert_array_equal(atom,atom_end)
    retained=start>0;left=int(np.sum(~retained));start=start[retained];end=end[retained]
    event=end<frames;steps=np.minimum(end,frames-1)-start
    counts=np.bincount(steps,minlength=frames);failures=np.bincount(steps[event],minlength=frames);risk=np.cumsum(counts[::-1])[::-1]
    survival=1.;area=0.;previous=0.;curve=[[0.,1.]];median=None
    for k in np.flatnonzero(counts):
        time=k*dt
        if time>tau+1e-12:break
        time=min(time,tau);area+=survival*(time-previous)
        survival*=1-failures[k]/risk[k];previous=time;curve.append([float(time),float(survival)])
        if median is None and survival<=.5:median=float(time)
    supported=bool(len(steps) and (max(steps)*dt>=tau-1e-12 or survival==0))
    if supported:area+=survival*(tau-previous);bounds=[float(area)]*2;estimate=float(area)
    else:bounds=[float(area),float(area+survival*(tau-previous))];estimate=None
    return dict(RMST_2ps=estimate,RMST_bounds_2ps=bounds,KM_survival=curve,KM_median_ps=median,
        episode_count=len(steps),completed_episodes=int(event.sum()),left_censored_excluded=left,right_censored=int((~event).sum()),
        one_frame_fraction_completed=float(np.mean(steps[event]==1)) if event.any() else None,
        completed_duration_mean_ps=float(np.mean(steps[event])*dt) if event.any() else None,
        dt_ps=dt,observation_span_ps=(frames-1)*dt)

def intermittent(masks,species,dt,lags):
    masks=np.asarray(masks,bool);species=np.asarray(species);result={}
    for time in lags:
        lag=round(time/dt)
        if lag<1 or not np.isclose(lag*dt,time,rtol=0,atol=1e-10) or lag>=len(masks):continue
        a,b=masks[:-lag],masks[lag:];den=a.sum()
        if not den:result[str(time)]=None;continue
        expected=sum(np.sum(a[:,species==s].sum(axis=1)*b[:,species==s].sum(axis=1))/np.sum(species==s) for s in np.unique(species))
        baseline=expected/den;observed=np.sum(a&b)/den
        result[str(time)]=float((observed-baseline)/(1-baseline)) if baseline<1 else None
    return result

def describe(masks,species,dt,lags,tau=2.):
    rows={}
    for key,selected in [('all',np.ones(len(species),bool)),('A',species==0),('Zr',species==1)]:
        m=masks[:,selected];s=species[selected]
        rows[key]={**lifetime(m,dt,tau), 'fraction':float(m.mean()),'intermittent':intermittent(m,s,dt,lags)}
    rows['balanced']=dict(RMST_bounds_2ps=np.mean([rows[k]['RMST_bounds_2ps'] for k in ['A','Zr']],axis=0).tolist(),
        definition='Equal mean of central-A and central-Zr RMST, within this independent replica')
    return rows

def replica_stats(bounds):
    a=np.asarray(bounds,float);assert a.shape==(3,2)
    return dict(mean_bounds=a.mean(axis=0).tolist(),SEM_bounds=(a.std(axis=0,ddof=1)/np.sqrt(3)).tolist(),replica_bounds=a.tolist())

def welch_difference(cu,ni):
    """Cu−Ni; expand the interval for unobserved survival-tail bounds."""
    a=np.asarray(cu,float);b=np.asarray(ni,float);assert a.shape==b.shape==(3,2)
    va=max(a.var(axis=0,ddof=1))/3;vb=max(b.var(axis=0,ddof=1))/3;se=float(np.sqrt(va+vb))
    df=float((va+vb)**2/(va*va/2+vb*vb/2)) if se else None
    margin=float(student_t.ppf(.975,df)*se) if se else 0.
    effect=[float(a[:,0].mean()-b[:,1].mean()),float(a[:,1].mean()-b[:,0].mean())]
    return dict(Cu_minus_Ni_bounds_ps=effect,SEM_ps=se,Welch_df=df,Welch_95_interval_ps=[effect[0]-margin,effect[1]+margin],
        Cu_over_Ni_ratio_bounds=[float(a[:,0].mean()/b[:,1].mean()),float(a[:,1].mean()/b[:,0].mean())] if b[:,0].mean()>0 else None,
        scope='Six independent replica means; no pooled episode/atom inference. Interval excludes model and finite-size systematics.')
