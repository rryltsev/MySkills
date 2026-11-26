"""Chemical bond-order features, train-only LDA and conditional motif diagnostics."""
import numpy as np
from scipy.special import sph_harm_y,expit
from .descriptors import neighbors

FEATURES=['CN','n_A','n_Zr','unlike_fraction','q4_all','q4_A','q4_Zr','q6_all','q6_A','q6_Zr','qbar4_all','qbar6_all']

def features(atoms,cutoff,elements,full=True):
    i,j,shifts,v=neighbors(atoms,cutoff);n=len(atoms);symbols=np.array(atoms.get_chemical_symbols())
    assert set(symbols)==set(elements)
    species=(symbols==elements[1]).astype(int);counts=np.column_stack([np.bincount(i[symbols[j]==s],minlength=n) for s in elements])
    cn=counts.sum(axis=1);assert np.all(cn>0)
    unlike=counts[np.arange(n),1-species]/cn
    r=np.linalg.norm(v,axis=1);theta=np.arccos(np.clip(v[:,2]/r,-1,1));phi=np.arctan2(v[:,1],v[:,0])
    columns=[cn,counts[:,0],counts[:,1],unlike];bars=[]
    for ell in ([4,6] if full else [6]):
        for channel in ([None,*elements] if full else [None]):
            mask=np.ones(len(i),bool) if channel is None else symbols[j]==channel
            number=np.bincount(i[mask],minlength=n);qlm=np.zeros((n,2*ell+1),complex)
            for k,m in enumerate(range(-ell,ell+1)):np.add.at(qlm[:,k],i[mask],sph_harm_y(ell,m,theta[mask],phi[mask]))
            qlm/=np.maximum(number,1)[:,None]
            q=np.sqrt(4*np.pi/(2*ell+1)*np.sum(abs(qlm)**2,axis=1));q[number==0]=np.nan
            columns.append(q)
            if full and channel is None:
                bar=qlm.copy();np.add.at(bar,i,qlm[j]);bar/=(cn+1)[:,None]
                bars.append(np.sqrt(4*np.pi/(2*ell+1)*np.sum(abs(bar)**2,axis=1)))
    x=np.column_stack(columns+bars) if full else np.column_stack([columns[0],columns[3],columns[-1]])
    # The analyzed cells have at most one image for each pair within this cutoff.
    heights=1/np.linalg.norm(np.linalg.inv(atoms.cell.array),axis=0);assert min(heights)>2*cutoff
    edges=np.column_stack([i[i<j],j[i<j]]);assert len(set(map(tuple,edges)))==len(edges)
    return x.astype(float),species,edges

def fit_lda(x,y):
    """Balanced two-class LDA; imputation/scaling/covariance use training data only."""
    x=np.asarray(x,float);y=np.asarray(y,int);assert set(y)=={0,1}
    med=np.array([np.nanmedian(c) if np.isfinite(c).any() else 0 for c in x.T]);missing=~np.isfinite(x)
    a=np.c_[np.where(missing,med,x),missing.astype(float)]
    center=a.mean(axis=0);scale=a.std(axis=0);scale[scale<1e-10]=1;z=(a-center)/scale
    means=np.array([z[y==k].mean(axis=0) for k in [0,1]])
    cov=sum(np.cov(z[y==k],rowvar=False) for k in [0,1])/2
    cov=.8*cov+.2*np.diag(np.diag(cov))+1e-6*np.eye(len(center))
    w=np.linalg.solve(cov,means[1]-means[0]);intercept=-.5*(means[1]+means[0])@w
    return dict(median=med,center=center,scale=scale,w=w,intercept=float(intercept))

def predict_lda(model,x):
    missing=~np.isfinite(x);a=np.c_[np.where(missing,model['median'],x),missing.astype(float)]
    return expit(((a-model['center'])/model['scale'])@model['w']+model['intercept'])

def permute_by_species(mask,species,rng):
    out=mask.copy()
    for s in np.unique(species):
        ids=np.flatnonzero(species==s);out[ids]=rng.permutation(mask[ids])
    return out

def cluster_metrics(mask,edges):
    ids=np.flatnonzero(mask);n=len(mask);hot=len(ids)
    if not hot:return np.array([0.,0.,0.])
    selected=edges[np.all(mask[edges],axis=1)];parent=np.arange(n);size=np.ones(n,int)
    def root(i):
        while parent[i]!=i:parent[i]=parent[parent[i]];i=parent[i]
        return i
    for i,j in selected:
        a,b=root(int(i)),root(int(j))
        if a!=b:
            if size[a]<size[b]:a,b=b,a
            parent[b]=a;size[a]+=size[b]
    sizes=np.unique([root(int(i)) for i in ids],return_counts=True)[1]
    return np.array([len(selected)/hot,max(sizes)/hot,np.sum(sizes*sizes)/hot])

def temporal_diagnostics(masks,species,dt=.2,tau=2.):
    """Species-conditioned intermittent correlation and censored continuous episodes."""
    masks=np.asarray(masks,bool);frames,n=masks.shape;correlation={}
    for lag in [1,2,5,10]:
        if lag>=frames:continue
        a,b=masks[:-lag],masks[lag:];den=a.sum();baseline=0.
        for s in np.unique(species):
            ids=species==s;baseline+=np.sum(a[:,ids].sum(axis=1)*b[:,ids].sum(axis=1))/np.sum(ids)
        baseline/=den;observed=np.sum(a&b)/den
        correlation[str(round(lag*dt,8))]=float((observed-baseline)/(1-baseline))
    durations=[];events=[];left=0
    for i in range(n):
        a=masks[:,i];starts=np.flatnonzero(a&~np.r_[False,a[:-1]])
        for start in starts:
            if start==0:left+=1;continue
            ends=np.flatnonzero(~a[start:]);event=bool(len(ends))
            duration=ends[0]*dt if event else (frames-1-start)*dt
            durations.append(float(duration));events.append(event)
    t=np.asarray(durations);e=np.asarray(events,bool);survival=1.;area=0.;previous=0.;curve=[[0.,1.]]
    for end in sorted(set(t[t<=tau])):
        area+=survival*(end-previous);risk=np.sum(t>=end);fail=np.sum((t==end)&e)
        if risk:survival*=1-fail/risk
        previous=end;curve.append([float(end),float(survival)])
    supported=bool(len(t) and (max(t)>=tau or survival==0))
    if supported:area+=survival*(tau-previous)
    return dict(intermittent_correlation=correlation,KM_RMST_2ps=area if supported else None,
        episode_count=len(t),right_censored=int(np.sum(~e)),left_censored_excluded=left,
        dt_ps=dt,observation_span_ps=(frames-1)*dt,KM_survival=curve,
        interpretation='Cadence-limited motif-label persistence, not an induction time or nucleation rate')
