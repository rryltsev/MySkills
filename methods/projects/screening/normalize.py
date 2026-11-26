"""Limited, deterministic AST normalization; no heuristic CAS simplification.

Constants are folded exactly as rational values, then emitted at round-trip
float64 precision. Raw-domain restrictions remain part of model semantics.
"""
import ast,math
from fractions import Fraction
import numpy as np
from .expressions import parse,evaluate,complexity

VERSION='canonical_ast_v2'

def N(value):return ('num',value if isinstance(value,Fraction) else Fraction(value))
def _key(n):return repr(n)
def _numeric(n):return n[0]=='num'

def _power(n,p):
    if p==0:return N(1)
    if p==1:return n
    if p<0:return _product([(n,p)])
    if _numeric(n):return N(n[1]**p)
    if n[0]=='pow':return _power(n[1],n[2]*p)
    if n[0]=='ratio':return _product([(n,p)])
    return ('pow',n,p)

def _product(items):
    coefficient=Fraction(1);powers={}
    def collect(node,exponent):
        nonlocal coefficient
        if _numeric(node):
            if node[1]==0 and exponent<0:raise ValueError('Constant division by zero')
            coefficient*=node[1]**exponent;return
        if node[0]=='ratio':
            c,num,den=node[1:]
            if c==0 and exponent<0:raise ValueError('Constant division by zero')
            coefficient*=c**exponent
            for v in num:collect(v,exponent)
            for v in den:collect(v,-exponent)
            return
        if node[0]=='pow':collect(node[1],node[2]*exponent);return
        powers[node]=powers.get(node,0)+exponent
    for node,e in items:collect(node,e)
    if coefficient==0:return N(0)
    num=[];den=[]
    for node,p in sorted(powers.items(),key=lambda t:_key(t[0])):
        if p>0:num.append(_power(node,p))
        if p<0:den.append(_power(node,-p))
    if not num and not den:return N(coefficient)
    if coefficient==1 and len(num)==1 and not den:return num[0]
    return ('ratio',coefficient,tuple(num),tuple(den))

def _add(items):
    terms={};constant=Fraction(0)
    def collect(node):
        nonlocal constant
        if node[0]=='add':
            for v in node[1]:collect(v)
        elif _numeric(node):constant+=node[1]
        else:
            if node[0]=='ratio':
                coefficient=node[1];base=_product([(N(1),1)]+[(v,1) for v in node[2]]+[(v,-1) for v in node[3]])
            else:coefficient=Fraction(1);base=node
            terms[base]=terms.get(base,Fraction(0))+coefficient
    for node in items:collect(node)
    result=[]
    if constant:result.append(N(constant))
    for base,c in sorted(terms.items(),key=lambda t:_key(t[0])):
        if c:result.append(_product([(N(c),1),(base,1)]))
    if not result:return N(0)
    if len(result)==1:return result[0]
    return ('add',tuple(result))

def _convert(n):
    if isinstance(n,ast.Constant):return N(Fraction.from_float(float(n.value)))
    if isinstance(n,ast.Name):return ('var',n.id)
    if isinstance(n,ast.UnaryOp):
        value=_convert(n.operand)
        return _product([(N(-1),1),(value,1)]) if isinstance(n.op,ast.USub) else value
    if isinstance(n,ast.BinOp):
        a,b=_convert(n.left),_convert(n.right)
        if isinstance(n.op,ast.Add):return _add([a,b])
        if isinstance(n.op,ast.Sub):return _add([a,_product([(N(-1),1),(b,1)])])
        if isinstance(n.op,ast.Mult):return _product([(a,1),(b,1)])
        if isinstance(n.op,ast.Div):return _product([(a,1),(b,-1)])
        if isinstance(n.op,ast.Pow):return _power(a,int(b[1]))
    if isinstance(n,ast.Call):
        value=_convert(n.args[0]);name=n.func.id
        if _numeric(value):
            answer={'sqrt':math.sqrt,'log':math.log,'exp':math.exp}[name](float(value[1]))
            if not math.isfinite(answer):raise ValueError('Nonfinite constant function')
            return N(Fraction.from_float(answer))
        return (name,value)
    raise ValueError('Unsupported AST')

def _num(n):
    value=float(n)
    if not math.isfinite(value):raise ValueError('Constant outside float64')
    return format(value,'.17g') if value else '0'

def _integer_power(base,p):
    if p==1:return base
    if p in (2,3):return f'(({base})**{p})'
    if not 1<=p<=1024:raise ValueError('Power too large')
    half=_integer_power(base,p//2);square=f'(({half})**2)'
    return square if p%2==0 else f'({square}*({base}))'

def _negative(n):
    return (_numeric(n) and n[1]<0) or (n[0]=='ratio' and n[1]<0)

def _positive(n):
    if _numeric(n):return N(-n[1])
    return ('ratio',-n[1],n[2],n[3])

def _emit(n):
    kind=n[0]
    if kind=='num':return _num(n[1])
    if kind=='var':return n[1]
    if kind in ('exp','log','sqrt'):return f'{kind}({_emit(n[1])})'
    if kind=='pow':return _integer_power(_emit(n[1]),n[2])
    if kind=='add':
        # Stable terms; subtraction has the same cost as addition.
        terms=list(n[1]);text=_emit(terms[0])
        for v in terms[1:]:text+=('-'+_emit(_positive(v))) if _negative(v) else ('+'+_emit(v))
        return '('+text+')'
    if kind=='ratio':
        coefficient,num,den=n[1:];parts=[_emit(v) for v in num]
        if abs(coefficient)!=1 or not parts:parts.insert(0,_num(coefficient))
        text='*'.join(parts) if parts else '1'
        if coefficient==-1 and num:text='(-('+text+'))'
        if den:text='('+text+')/('+'*'.join(_emit(v) for v in den)+')'
        return '('+text+')'
    raise ValueError(kind)

def normalize(raw):
    raw=raw.strip().replace('^','**');current=raw
    # Folding can make two constants coincide at float64 precision. Reach the
    # fixed point explicitly, rather than expose a non-idempotent formatter.
    seen=set()
    for _ in range(8):
        with np.errstate(all='ignore'):canonical=_emit(_convert(parse(current)))
        parse(canonical)
        if canonical==current:break
        if canonical in seen:raise ValueError('Normalization cycle')
        seen.add(current);current=canonical
    else:raise ValueError('Normalization did not reach a fixed point')
    return dict(raw_expression=raw,canonical_expression=canonical,
        complexity_raw=complexity(raw),complexity=complexity(canonical),normalization_version=VERSION)

def model_predict(model,x):
    raw,_=evaluate(model['raw_expression'],x);y,_=evaluate(model['canonical_expression'],x)
    y[~np.isfinite(raw)]=np.nan
    return y

def verify_equivalence(model,x,sy):
    raw,_=evaluate(model['raw_expression'],x);can,_=evaluate(model['canonical_expression'],x)
    if not np.isfinite(raw).all() or not np.isfinite(can).all():return False
    return bool(np.allclose(raw,can,rtol=1e-9,atol=1e-10*max(float(sy),1e-12)))
