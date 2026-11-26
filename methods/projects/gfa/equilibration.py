"""Conservative diagnostics for fixed-volume, unwrapped MD trajectories."""
import numpy as np

def block_summary(values,block_size):
    x=np.asarray(values,float);n=len(x)//block_size
    if n<4:raise ValueError('At least four blocks required')
    means=x[:n*block_size].reshape(n,block_size).mean(axis=1)
    return dict(mean=float(np.mean(x)),std=float(np.std(x,ddof=1)),block_count=n,block_mean_sem=float(np.std(means,ddof=1)/np.sqrt(n)),half_difference=float(np.mean(x[len(x)//2:])-np.mean(x[:len(x)//2])))

def motion(positions,masses,species,dt_ps,k_Ainv,max_lag=None):
    x=np.asarray(positions,float);mass=np.asarray(masses)
    # Fixed-volume only. Remove mass-weighted COM, never wrap a displacement.
    x=x-(x*mass[None,:,None]).sum(axis=1,keepdims=True)/mass.sum()
    n=len(x);max_lag=min(max_lag or n//2,n-1)
    lags=np.unique(np.r_[0,np.arange(1,min(101,max_lag+1)),np.linspace(100,max_lag,100).astype(int)])
    lags=lags[(lags>=0)&(lags<=max_lag)]
    result={s:dict(time_ps=[],msd_A2=[],self_isf=[]) for s in sorted(set(species))}
    for lag in lags:
        dr=x[lag:]-x[:n-lag] if lag else np.zeros_like(x)
        for s,row in result.items():
            d2=np.sum(dr[:,np.array(species)==s,:]**2,axis=-1)
            row['time_ps'].append(float(lag*dt_ps));row['msd_A2'].append(float(d2.mean()))
            row['self_isf'].append(float(np.sinc(k_Ainv*np.sqrt(d2)/np.pi).mean()))
    for row in result.values():
        t=np.array(row['time_ps']);f=np.array(row['self_isf']);msd=np.array(row['msd_A2'])
        hit=np.where(f<=np.exp(-1))[0]
        row['tau_1e_ps']=float(np.interp(np.exp(-1),f[hit[0]-1:hit[0]+1][::-1],t[hit[0]-1:hit[0]+1][::-1])) if len(hit) and hit[0]>0 else None
        fit=(t>=2)&(t<=5)
        slope,intercept=np.polyfit(t[fit],msd[fit],1)
        row['D_2to5ps_A2_ps']=float(slope/6)
        row['diffusion_fit_intercept_A2']=float(intercept)
        if np.all(msd[fit]>0):
            row['MSD_log_slope_2to5ps']=float(np.polyfit(np.log(t[fit]),np.log(msd[fit]),1)[0])
        else:row['MSD_log_slope_2to5ps']=None
        for lower,upper in ((1,3),(3,5)):
            mask=(t>=lower)&(t<=upper)
            row[f'D_{lower}to{upper}ps_A2_ps']=float(np.polyfit(t[mask],msd[mask],1)[0]/6)
        row['msd_at_5ps_A2']=float(np.interp(5,t,msd))
        row['Fs_at_5ps']=float(np.interp(5,t,f))
    return result
