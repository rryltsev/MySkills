import math
import numpy as np
from . import formula_tools as ft


def integer_power(base,k):
    """Preserve integer-power meaning using only public **2, **3, *, / operators."""
    if not isinstance(k,int) or abs(k)>32:raise ValueError('Integer exponent outside validated export range')
    if k<0:return '(1/('+integer_power(base,-k)+'))'
    if k==0:return '1'
    if k==1:return base
    if k in (2,3):return '(('+base+')**'+str(k)+')'
    if k%2==0:return '('+integer_power(base,k//2)+'**2)'
    return '('+base+'*'+integer_power(base,k-1)+')'


def sympy_string(s):
    import sympy as sp
    if s.is_Number:
        if s.is_Rational and not s.is_Integer:
            # Exact binary coefficients often have large integer numerators
            # and denominators, which are forbidden public numeric leaves.
            # Emit a float literal only when its binary value is EXACTLY the
            # same rational. Other rationals retain the original checked path.
            try:
                value=float(s)
            except (OverflowError,ValueError):
                value=float('inf')
            if math.isfinite(value) and sp.Rational(*value.as_integer_ratio())==s:
                return repr(value)
            return '('+str(s.p)+'/'+str(s.q)+')'
        return str(s)
    if s.is_Symbol:
        if str(s) not in ft.VARIABLES:raise ValueError(str(s))
        return str(s)
    args=[sympy_string(x) for x in s.args]
    if s.func==sp.Add:return '('+'+'.join(args)+')'
    if s.func==sp.Mul:return '('+'*'.join(args)+')'
    if s.func==sp.Pow:
        if s.args[1].is_Integer:return integer_power(args[0],int(s.args[1]))
        if s.args[1]==sp.Rational(1,2):return 'sqrt('+args[0]+')'
        raise ValueError('Non-grammar exponent')
    if s.func in [sp.sin,sp.cos,sp.exp,sp.log]:return str(s.func)+'('+args[0]+')'
    raise ValueError('Unsupported SymPy operator '+str(s.func))


def op_records(tree,names):
    records=[]
    for n in tree.Nodes:
        rec={'type':str(n.Type).split('.')[-1].lower(),'value':float(n.Value),'arity':int(n.Arity),'length':int(n.Length),'hash':int(n.HashValue)}
        if n.IsVariable:rec['variable']=names[int(n.HashValue)]
        if not n.IsLeaf and float(n.Value)!=1.:raise ValueError('Unvalidated non-leaf coefficient')
        records.append(rec)
    return records


def op_replay(records,X,dtype=np.float64):
    X=np.asarray(X,dtype=dtype)
    def rec(i):
        n=records[i];kind=n['type'];value=dtype(n['value'])
        if kind=='constant':return np.full(len(X),value,dtype=dtype)
        if kind=='variable':return np.asarray(value*X[:,ft.VARIABLES.index(n['variable'])],dtype=dtype)
        children=[];j=i-1
        for _ in range(n['arity']):children.append(rec(j));j-=records[j]['length']+1
        if kind in ['add','mul','sub','div']:
            out=children[0]
            for a in children[1:]:
                if kind=='add':out=out+a
                elif kind=='mul':out=out*a
                elif kind=='sub':out=out-a
                else:out=out/a
            return np.asarray(out,dtype=dtype)
        if kind=='square':return np.asarray(children[0]*children[0],dtype=dtype)
        if kind=='pow':return np.asarray(np.power(children[0],children[1]),dtype=dtype)
        fun={'sin':np.sin,'cos':np.cos,'exp':np.exp,'log':np.log,'sqrt':np.sqrt}.get(kind)
        if fun is None:raise ValueError('Unvalidated native primitive '+kind)
        return np.asarray(fun(children[0]),dtype=dtype)
    with np.errstate(all='raise',under='ignore'):return rec(len(records)-1)


def op_expression(records):
    def rec(i):
        n=records[i];kind=n['type']
        if kind=='constant':return repr(n['value'])
        if kind=='variable':return '('+repr(n['value'])+'*'+n['variable']+')'
        args=[];j=i-1
        for _ in range(n['arity']):args.append(rec(j));j-=records[j]['length']+1
        if kind in ['add','sub','mul','div']:return '('+{'add':'+','sub':'-','mul':'*','div':'/'}[kind].join(args)+')'
        if kind=='square':return '('+args[0]+'**2)'
        if kind=='pow':
            exponent=records[i-1-records[i-1]['length']-1]
            if exponent['type']!='constant' or not float(exponent['value']).is_integer():raise ValueError('Noninteger power')
            return integer_power(args[0],int(exponent['value']))
        if kind not in ['sin','cos','exp','log','sqrt']:raise ValueError(kind)
        return kind+'('+args[0]+')'
    return rec(len(records)-1)


def scaled_error(a,b):return float(np.max(np.abs(a-b)/(1+np.abs(a)+np.abs(b))))
