"""Multi-origin relaxation, threshold-free correlations and atom-identity retention."""
import numpy as np
from scipy.fft import rfft,irfft,next_fast_len
from .motif_lifetimes import lifetime

def first_crossing(times,curve,level=np.exp(-1)):
    t=np.asarray(times);c=np.asarray(curve);hits=np.flatnonzero(c<=level)
    if not len(hits):return dict(time_ps=None,bracket_ps=None,first_interval=False)
    k=int(hits[0])
    if k==0:return dict(time_ps=float(t[0]),bracket_ps=[float(t[0])]*2,first_interval=True)
    fraction=(c[k-1]-level)/(c[k-1]-c[k])
    return dict(time_ps=float(t[k-1]+fraction*(t[k]-t[k-1])),bracket_ps=[float(t[k-1]),float(t[k])],first_interval=k==1)

def multi_fs(x,masses,species,dt,ks,maxlag):
    x=np.asarray(x,float).copy();x-=np.average(x,axis=1,weights=masses)[:,None,:]
    fs=np.ones((len(ks),2,maxlag+1));msd=np.zeros((2,maxlag+1))
    for lag in range(1,maxlag+1):
        d=x[lag:]-x[:-lag];radius=np.linalg.norm(d,axis=2)
        for s in [0,1]:
            r=radius[:,species==s];msd[s,lag]=np.mean(r*r)
            for k,q in enumerate(ks):fs[k,s,lag]=np.sinc(q*r/np.pi).mean()
    return fs,msd

def autocorrelation(x,maxlag,frame_centered=True):
    x=np.asarray(x,float).copy();assert x.ndim==2 and maxlag<len(x)
    x-=x.mean(axis=1,keepdims=True) if frame_centered else x.mean()
    n=len(x);size=next_fast_len(2*n-1);f=rfft(x,n=size,axis=0)
    summed=irfft(np.sum(abs(f)**2,axis=1),n=size)[:maxlag+1]
    variance=np.mean(x*x)
    if variance<=1e-20:raise ValueError('Correlation undefined for zero variance')
    return summed/((n-np.arange(maxlag+1))*x.shape[1]*variance)

def curve_summary(times,curve):
    times=np.asarray(times);curve=np.asarray(curve)
    return dict(crossing_1e=first_crossing(times,curve),signed_integrals_ps={str(end):float(np.trapezoid(curve[times<=end+1e-12],times[times<=end+1e-12])) for end in [1.,2.,4.] if times[-1]>=end-1e-12})

def hysteresis(bands,entry_band,exit_band):
    assert entry_band>=exit_band
    state=np.empty(bands.shape,bool);state[0]=bands[0]>=entry_band
    for t in range(1,len(bands)):state[t]=(bands[t]>=entry_band)|(state[t-1]&(bands[t]>=exit_band))
    return state

def bond_diagnostics(state,pairs,species,origins,maxlag,dt,cohort_masks):
    """State is on the analysis window; complete pre-window hysteresis is a caller duty."""
    n=len(species);kind=species[pairs[:,0]]+species[pairs[:,1]];times=np.arange(maxlag+1)*dt
    names=['AA','A-Zr','Zr-Zr'];survive=np.zeros((3,maxlag+1));returns=survive.copy();den=np.zeros(3)
    cohort_keys=[(m,s,hot,channel) for m in cohort_masks for s in [0,1] for hot in [True,False] for channel in ['all','unlike']]
    cs=np.zeros((len(cohort_keys),maxlag+1));cr=cs.copy();cd=np.zeros(len(cohort_keys))
    for origin in origins:
        ids=np.flatnonzero(state[origin]);p=pairs[ids];kk=kind[ids];alive=np.ones(len(ids),bool)
        den+=np.bincount(kk,minlength=3)
        base={}
        for channel,select in [('all',np.ones(len(ids),bool)),('unlike',kk==1)]:
            base[channel]=np.bincount(p[select,0],minlength=n)+np.bincount(p[select,1],minlength=n)
        selected=[]
        for j,(motif,s,hot,channel) in enumerate(cohort_keys):
            use=(species==s)&(cohort_masks[motif][origin]==hot)&(base[channel]>0)
            selected.append(use);cd[j]+=use.sum()
        for lag in range(maxlag+1):
            present=state[origin+lag,ids];alive&=present
            survive[:,lag]+=np.bincount(kk[alive],minlength=3);returns[:,lag]+=np.bincount(kk[present],minlength=3)
            fractions={}
            for channel,select in [('all',np.ones(len(ids),bool)),('unlike',kk==1)]:
                for key,flag in [('continuous',alive),('return',present)]:
                    q=p[select&flag];counts=np.bincount(q[:,0],minlength=n)+np.bincount(q[:,1],minlength=n)
                    fractions[channel,key]=counts/np.maximum(base[channel],1)
            for j,(_,_,_,channel) in enumerate(cohort_keys):
                cs[j,lag]+=fractions[channel,'continuous'][selected[j]].sum();cr[j,lag]+=fractions[channel,'return'][selected[j]].sum()
    pair_rows={}
    na=np.sum(species==0);nz=np.sum(species==1);possible=np.array([na*(na-1)/2,na*nz,nz*(nz-1)/2])
    for k,name in enumerate(names):
        s=survive[k]/den[k];b=returns[k]/den[k];baseline=float(np.mean(state[:,kind==k].sum(axis=1))/possible[k])
        km=lifetime(state[:,kind==k],dt)
        connected=((b-baseline)/(1-baseline)).tolist() if baseline<1 else None
        pair_rows[name]=dict(continuous=s.tolist(),intermittent=b.tolist(),intermittent_connected=connected,
            baseline_pair_occupancy=baseline,initial_pair_occurrences=int(den[k]),
            residual_RMST_2ps=float(np.sum(s[:-1])*dt),entry_episodes=km)
    cohorts=[]
    for j,(motif,s,hot,channel) in enumerate(cohort_keys):
        assert cd[j]>0
        a=cs[j]/cd[j];b=cr[j]/cd[j]
        cohorts.append(dict(motif=motif,central_species=['A','Zr'][s],initial_high_label=hot,neighbor_channel=channel,
            initial_center_occurrences=int(cd[j]),continuous=a.tolist(),intermittent=b.tolist(),
            residual_integral_2ps=float(np.sum(a[:-1])*dt)))
    return dict(times_ps=times.tolist(),pair_types=pair_rows,cohorts=cohorts,origins=len(origins))
