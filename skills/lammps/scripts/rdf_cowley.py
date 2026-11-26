"""NumPy-only continuous RDF minima, shell counts and Cowley estimators.

RDF points are bin centres; cumulative coordination points are UPPER bin edges.
Those coordinates are intentionally separate. Statistical units are trajectories.
"""
import numpy as np

def continuous_minimum(r, g, minimum_window, half_window=0.175, degree=4):
    """Refine a bracketed FIRST-shell trough by a local polynomial root.

    The caller inspects the first peak and supplies the following trough bracket.
    fit_SE is a local fit diagnostic, not a trajectory confidence interval.
    """
    r=np.asarray(r,float);g=np.asarray(g,float)
    if r.ndim!=1 or r.size==0 or r.shape!=g.shape or not np.isfinite(r).all() or np.any(np.diff(r)<=0):raise ValueError('Invalid radial grid')
    minimum_window=np.asarray(minimum_window,float)
    if minimum_window.shape!=(2,) or not np.isfinite(minimum_window).all() or minimum_window[0]>=minimum_window[1]:raise ValueError('Need an increasing finite minimum bracket')
    if not np.isfinite(half_window) or half_window<=0 or not isinstance(degree,(int,np.integer)) or degree<2:raise ValueError('Need a positive fit half-window and integer degree >=2')
    use=np.isfinite(g)&(r>=minimum_window[0])&(r<=minimum_window[1])
    ids=np.flatnonzero(use)
    if len(ids)<max(7,degree+3):raise ValueError('Insufficient finite bins in the minimum bracket')
    # Local polynomial filtering is only used to locate the trough.
    coarse=[]
    for k in ids:
        sel=np.isfinite(g)&(abs(r-r[k])<=half_window+1e-12)
        if sel.sum()<degree+3:coarse.append(np.inf);continue
        t=(r[sel]-r[k])/half_window
        cc=np.polynomial.polynomial.polyfit(t,g[sel],degree)
        coarse.append(cc[0])
    k=ids[int(np.argmin(coarse))];seed=r[k]
    if k in [ids[0],ids[-1]]:raise ValueError('Minimum touches bracket edge; inspect the RDF')
    sel=np.isfinite(g)&(abs(r-seed)<=half_window+1e-12);t=(r[sel]-seed)/half_window
    design=np.polynomial.polynomial.polyvander(t,degree);coef=np.linalg.lstsq(design,g[sel],rcond=None)[0]
    roots=np.polynomial.polynomial.polyroots(np.polynomial.polynomial.polyder(coef))
    candidates=[]
    for z in roots:
        if abs(z.imag)>1e-8 or abs(z.real)>.65:continue
        rr=seed+z.real*half_window
        curvature=np.polynomial.polynomial.polyval(z.real,np.polynomial.polynomial.polyder(coef,2))/half_window**2
        if curvature>0 and minimum_window[0]<rr<minimum_window[1]:candidates.append((abs(rr-seed),rr,z.real,curvature))
    if not candidates:raise ValueError('No positive-curvature interior minimum; do not substitute an argmin bin')
    _,radius,tmin,curvature=min(candidates);residual=g[sel]-design@coef;rss=residual@residual;dof=max(sel.sum()-degree-1,1)
    covariance=np.linalg.pinv(design.T@design)*(rss/dof)
    gradient=np.array([0]+[j*tmin**(j-1) for j in range(1,degree+1)])/half_window
    local_se=np.sqrt(max(gradient@covariance@gradient,0))/curvature
    rms=np.sqrt(np.mean(residual**2));resolution=curvature*half_window**2/max(rms,1e-15)
    return {'radius':float(radius),'g_min':float(np.polynomial.polynomial.polyval(tmin,coef)),
            'curvature':float(curvature),'fit_RMS':float(rms),'fit_SE_diagnostic':float(local_se),
            'resolution_ratio':float(resolution),'weak_minimum':bool(resolution<3),
            'half_window':float(half_window),'degree':degree,'coarse_bin_center':float(seed)}

def regularize_minima(x, radii, penalties=None):
    """Blind replica-CV/1SE second-derivative smoothing at measured x points.

    radii is (independent trajectories, compositions). Returns its linear map
    for conditional resampling with fixed weights/penalty. For full uncertainty,
    rerun selection on each replica resample with shell counts attached.
    No expected critical composition is an input to this estimator.
    """
    x=np.asarray(x,float);y=np.asarray(radii,float)
    if x.ndim!=1 or not np.isfinite(x).all() or y.ndim!=2 or y.shape[1]!=len(x) or len(x)<4 or not np.isfinite(y).all():raise ValueError('Need finite replica radii on >=4 compositions')
    if np.any(np.diff(x)<=0) or y.shape[0]<3:raise ValueError('Need increasing x and >=3 independent replicas')
    u=(x-x[0])/(x[-1]-x[0]);dx=np.diff(u);n=len(x);dd=np.zeros((n-2,n))
    for j in range(n-2):dd[j,j:j+3]=2/(dx[j]+dx[j+1])*np.array([1/dx[j],-1/dx[j]-1/dx[j+1],1/dx[j+1]])
    penalty=(dd.T*((dx[:-1]+dx[1:])/2))@dd;penalty/=np.diag(penalty).mean()
    sd=y.std(0,ddof=1);scale=np.maximum(sd,max(np.median(sd)*.25,1e-12));w=1/scale**2;w/=np.median(w);w=np.clip(w,.1,10)
    penalties=np.logspace(-4,8,41) if penalties is None else np.asarray(penalties,float)
    if penalties.ndim!=1 or penalties.size==0 or not np.isfinite(penalties).all() or np.any(penalties<0):raise ValueError('Need finite nonnegative smoothing penalties')
    penalties=np.unique(penalties)
    losses=[];maps=[];nrep=len(y);train=(y.sum(0)[None,:]-y)/(nrep-1)
    for lam in penalties:
        h=np.linalg.solve(np.diag(w)+lam*penalty,np.diag(w));pred=(h@train.T).T
        losses.append(np.mean(((pred-y)/scale)**2,axis=1));maps.append(h)
    losses=np.asarray(losses);cv=losses.mean(1);se=losses.std(1,ddof=1)/np.sqrt(nrep);best=np.argmin(cv);selected=np.flatnonzero(cv<=cv[best]+se[best])[-1]
    h=maps[selected];alternatives=np.array([np.linalg.solve(np.diag(w)+factor*penalties[selected]*penalty,np.diag(w))@y.mean(0) for factor in [1/3,1,3]])
    return {'radius':h@y.mean(0),'map':h,'lambda':float(penalties[selected]),'lambda_min':float(penalties[best]),'effective_df':float(np.trace(h)),'penalty_variants':alternatives,'CV_loss':cv,'CV_SE':se,'weights':w}

def cumulative_at_radius(edges, cumulative, radius):
    """Continuous interpolation of counted coordination at UPPER bin edges.

    Assumes the stored counts start at r=0 and adds its exact zero. This avoids
    rounding the cutoff back to a bin. All-NaN channels remain undefined;
    partially missing, negative or decreasing cumulative counts are rejected.
    Interpolation resolves integration-boundary quantization; it cannot recover
    unknown sub-bin pair distances. Check sensitivity to radial bin width.
    """
    edges=np.asarray(edges,float);z=np.asarray(cumulative,float)
    if edges.ndim!=1 or edges.size==0 or z.ndim<1 or z.shape[-1]!=len(edges) or not np.isfinite(edges).all() or np.any(np.diff(edges)<=0) or edges[0]<=0:raise ValueError('Cumulative values require positive finite upper bin edges')
    valid=np.isfinite(z).all(axis=-1)|np.isnan(z).all(axis=-1)
    if not np.all(valid) or np.any(z<0) or np.any(np.diff(z,axis=-1)<-1e-12*np.maximum(1,abs(z[...,:-1]))):raise ValueError('Need nonnegative nondecreasing counts or all-NaN channels')
    if not 0<=radius<=edges[-1]:raise ValueError('Cutoff outside stored radial range')
    k=int(np.searchsorted(edges,radius));lo=0. if k==0 else edges[k-1];left=np.zeros(z.shape[:-1]) if k==0 else z[...,k-1]
    if k==len(edges):return z[...,-1].copy()
    return left+(z[...,k]-left)*((radius-lo)/(edges[k]-lo))

def cowley(coordination, populations, finite_population=False):
    """Directed alpha_ab=1-Zab/(Za*p(b|a)); last axes are species a,b.

    populations are actual nonnegative integer species counts, not fractions.
    A missing channel for a present species makes the entire center row NaN;
    it must not be silently omitted from total coordination. Absent neighbor
    species have exactly zero counts and undefined alpha.
    """
    z=np.asarray(coordination,float);counts=np.asarray(populations,float)
    if counts.ndim!=1 or counts.size==0 or not np.isfinite(counts).all() or np.any(counts<0) or np.any(counts!=np.floor(counts)):raise ValueError('Need finite nonnegative integer populations')
    n=counts.sum();ns=len(counts)
    if z.ndim<2 or z.shape[-2:]!=(ns,ns) or n<=1 or np.isinf(z).any() or np.any(z<0):raise ValueError('Invalid coordination/populations')
    if np.any(z[...,counts==0,:]>0) or np.any(z[...,:,counts==0]>0):raise ValueError('Absent species cannot have positive coordination')
    z=z.copy();z[...,:,counts==0]=0
    total=np.sum(z,axis=-1)[...,None]
    fractions=np.divide(z,total,out=np.full_like(z,np.nan),where=np.isfinite(total)&(total>0))
    baseline=(counts[None,:]-np.eye(ns))/(n-1) if finite_population else np.broadcast_to(counts/n,(ns,ns))
    out=1-np.divide(fractions,baseline,out=np.full_like(z,np.nan),where=baseline>0)
    out[...,counts==0,:]=np.nan;out[...,:,counts==0]=np.nan
    return out

def affine_background(x, values, fit_mask=None):
    """OLS line a+b*x along the last axis; returns fit and residual.

    Default: use the entire available composition interval. The user can pass a
    sideband mask as a separately named sensitivity analysis.
    """
    x=np.asarray(x,float);y=np.asarray(values,float)
    if x.ndim!=1 or not np.isfinite(x).all() or y.ndim<1 or y.shape[-1]!=x.size or np.unique(x).size<2:raise ValueError('Need finite distinct x and matching values')
    mask=np.isfinite(y).all(axis=tuple(range(y.ndim-1))) if y.ndim>1 else np.isfinite(y)
    if fit_mask is not None:
        fit_mask=np.asarray(fit_mask,bool)
        if fit_mask.shape!=x.shape:raise ValueError('Fit mask must match x')
        mask&=fit_mask
    if mask.sum()<2:raise ValueError('Insufficient points for an affine fit')
    if np.unique(x[mask]).size<2:raise ValueError('Need at least two distinct fit x values')
    design=np.column_stack([np.ones(len(x)),x]);co=np.linalg.lstsq(design[mask],y[...,mask].reshape(-1,mask.sum()).T,rcond=None)[0]
    fitted=(design@co).T.reshape(y.shape);return fitted,y-fitted,co.T.reshape(y.shape[:-1]+(2,))

def _self_check():
    r=(np.arange(240)+.5)*.025;true=3.8137;g=.6+.8*(r-true)**2+.12*(r-true)**3
    q=continuous_minimum(r,g,(3.4,4.2),degree=4);np.testing.assert_allclose(q['radius'],true,atol=1e-10)
    edges=np.arange(1,101)*.1;z=edges**2;np.testing.assert_allclose(cumulative_at_radius(edges,z,3.45),.5*(3.4**2+3.5**2))
    populations=np.array([970,30]);p=(populations[None,:]-np.eye(2))/(populations.sum()-1);zz=np.array([12.,10.])[:,None]*p
    np.testing.assert_allclose(cowley(zz,populations,True),0,atol=1e-14)
    assert np.isnan(cowley(np.array([[12.,0.],[np.nan,np.nan]]),[1000,0])[:,1]).all()
    assert np.isnan(cowley([[2.,np.nan],[1.,3.]],[5,5])[0]).all()
    assert np.isnan(cowley(np.zeros((2,2)),[5,5])).all()
    assert np.isnan(cowley([[0.,4.],[.25,3.75]],[1,4],True)[0,0])
    assert np.isnan(cumulative_at_radius([1,2],[np.nan,np.nan],1.5))
    for invalid in [lambda: cowley([[-1,2],[2,1]],[5,5]),
                    lambda: cowley([[1,2],[2,1]],[5.5,4.5]),
                    lambda: cumulative_at_radius([1,2],[2,1],1.5),
                    lambda: cumulative_at_radius([],[],0)]:
        try: invalid()
        except ValueError: pass
        else: raise AssertionError('Invalid scientific input was accepted')
    x=np.linspace(0,.06,31);y=2+3*x;_,res,_=affine_background(x,y);np.testing.assert_allclose(res,0,atol=2e-15)
    print('PASS: analytic minimum, sub-bin CDF, random-label Cowley, absent/singleton species, missing channels, zero coordination, invalid inputs, affine residual')

if __name__=='__main__':_self_check()
