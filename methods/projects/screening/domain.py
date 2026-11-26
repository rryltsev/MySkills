"""Public-domain screening independent of validation labels and test inputs.

Conservative interval arithmetic + bisection, with numeric probes.
This is an engineering filter, not a formally verified interval library.
Unresolved is distinct from an observed domain failure.
"""
import ast
import numpy as np
from scipy.stats import qmc
from .expressions import parse,evaluate
from .normalize import verify_equivalence

def probes(bounds,grid_n=65,sobol_n=1024,sobol_seed=71129):
    b=np.asarray(bounds,float);g=np.meshgrid(*[np.linspace(a,z,grid_n) for a,z in b],indexing='ij')
    grid=np.column_stack([v.ravel() for v in g])
    u=qmc.Sobol(2,scramble=True,seed=sobol_seed).random_base2(int(np.log2(sobol_n)))
    if len(u)!=sobol_n:raise ValueError('sobol_n must be a power of 2')
    return np.vstack([grid,qmc.scale(u,b[:,0],b[:,1])])

def _out(lo,hi):
    return np.nextafter(lo,-np.inf),np.nextafter(hi,np.inf)

def _interval(n,box):
    size=len(box);yes=np.ones(size,dtype=bool)
    if isinstance(n,ast.Constant):
        a=np.full(size,float(n.value));return a,a,yes
    if isinstance(n,ast.Name):return box[:,int(n.id[1])-1,0],box[:,int(n.id[1])-1,1],yes
    if isinstance(n,ast.UnaryOp):
        lo,hi,ok=_interval(n.operand,box)
        return (-hi,-lo,ok) if isinstance(n.op,ast.USub) else (lo,hi,ok)
    if isinstance(n,ast.Call):
        lo,hi,ok=_interval(n.args[0],box);name=n.func.id
        if name=='exp':
            lo,hi=_out(np.exp(lo),np.exp(hi));lo=np.maximum(lo,0)
        elif name=='log':
            ok=ok&(lo>0);lo,hi=_out(np.log(lo),np.log(hi))
        else:
            ok=ok&(lo>=0);lo,hi=_out(np.sqrt(lo),np.sqrt(hi));lo=np.maximum(lo,0)
        return lo,hi,ok&np.isfinite(lo)&np.isfinite(hi)
    lo,hi,ok=_interval(n.left,box)
    if isinstance(n.op,ast.Pow):
        power=int(ast.literal_eval(n.right))
        if power==2:
            a=np.where((lo<=0)&(hi>=0),0,np.minimum(lo*lo,hi*hi));b=np.maximum(lo*lo,hi*hi)
            a,b=_out(a,b);a=np.maximum(a,0)
        else:a,b=_out(lo**3,hi**3)
        return a,b,ok&np.isfinite(a)&np.isfinite(b)
    a,b,valid=_interval(n.right,box);ok=ok&valid
    if isinstance(n.op,ast.Mult) and ast.dump(n.left)==ast.dump(n.right):
        l=np.where((lo<=0)&(hi>=0),0,np.minimum(lo*lo,hi*hi));h=np.maximum(lo*lo,hi*hi)
        l,h=_out(l,h);l=np.maximum(l,0)
        return l,h,ok&np.isfinite(l)&np.isfinite(h)
    if isinstance(n.op,ast.Add):l,h=_out(lo+a,hi+b)
    elif isinstance(n.op,ast.Sub):l,h=_out(lo-b,hi-a)
    else:
        if isinstance(n.op,ast.Div):
            ok=ok&((a>0)|(b<0));a,b=_out(1/b,1/a)
        values=np.array([lo*a,lo*b,hi*a,hi*b]);l,h=_out(np.min(values,axis=0),np.max(values,axis=0))
    return l,h,ok&np.isfinite(l)&np.isfinite(h)

def interval_check(expression,bounds,max_depth=12):
    tree=parse(expression);boxes=np.asarray(bounds,float)[None,:,:];count=0
    widths=boxes[0,:,1]-boxes[0,:,0]
    with np.errstate(all='ignore'):
        for depth in range(max_depth+1):
            count+=len(boxes);_,_,valid=_interval(tree,boxes)
            boxes=boxes[~valid]
            if not len(boxes):return dict(status='passed',cells_checked=count,depth=depth)
            if depth==max_depth:return dict(status='unresolved',cells_checked=count,depth=depth,unresolved_cells=len(boxes))
            # Alternating dimensions: split longest side in normalized coordinates.
            relative=(boxes[:,:,1]-boxes[:,:,0])/widths
            axes=np.argmax(relative,axis=1);left=boxes.copy();right=boxes.copy()
            for i,axis in enumerate(axes):
                mid=(boxes[i,axis,0]+boxes[i,axis,1])/2
                left[i,axis,1]=mid;right[i,axis,0]=mid
            boxes=np.concatenate([left,right])
    raise AssertionError('unreachable')

def check(model,bounds,cfg,sy):
    x=probes(bounds,**{k:cfg[k] for k in ['grid_n','sobol_n','sobol_seed']})
    raw,_=evaluate(model['raw_expression'],x);can,_=evaluate(model['canonical_expression'],x)
    bad=int((~(np.isfinite(raw)&np.isfinite(can))).sum())
    if bad:return dict(status='failed',reason='nonfinite_probes',bad_probe_count=bad,n_probes=len(x))
    if not verify_equivalence(model,x,sy):return dict(status='failed',reason='normalization_mismatch',n_probes=len(x))
    audits={name:interval_check(model[name+'_expression'],bounds,cfg['max_depth']) for name in ['raw','canonical']}
    return dict(status='passed' if all(r['status']=='passed' for r in audits.values()) else 'unresolved',n_probes=len(x),interval=audits)
