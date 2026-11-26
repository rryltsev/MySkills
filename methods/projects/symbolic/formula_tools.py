"""Public formula grammar and score, SR-HACK-2026-v1.

Only NumPy is required. No eval/exec or arbitrary Python is used.
Complexity is measured on the submitted tree after constant folding only.
"""
import argparse
import ast
import json
import math
from dataclasses import dataclass
from pathlib import Path
import numpy as np

VARIABLES = ('x1', 'x2', 'x3')
FUNCTIONS = {'sin': np.sin, 'cos': np.cos, 'exp': np.exp,
             'log': np.log, 'sqrt': np.sqrt}
WEIGHTS = {'add': 1, 'sub': 1, 'mul': 1, 'div': 2,
           'neg': 1, 'pow2': 1, 'pow3': 1,
           'sin': 3, 'cos': 3, 'exp': 3, 'log': 3, 'sqrt': 3}
ERROR_BAD, ERROR_GOOD, COMPLEXITY_SCALE = 1.0, 0.001, 100.0
MAX_COMPLEXITY, MAX_DEPTH = 120, 24

@dataclass(frozen=True)
class Node:
    op: str
    args: tuple = ()
    value: object = None

    @property
    def complexity(self):
        return (1 if self.op in ('number', 'variable') else
                WEIGHTS[self.op] + sum(a.complexity for a in self.args))

    def evaluate(self, X):
        if self.op == 'number':
            return float(self.value)
        if self.op == 'variable':
            return X[:, VARIABLES.index(self.value)]
        a = [z.evaluate(X) for z in self.args]
        if self.op == 'add': return a[0] + a[1]
        if self.op == 'sub': return a[0] - a[1]
        if self.op == 'mul': return a[0] * a[1]
        if self.op == 'div': return np.divide(a[0], a[1])
        if self.op == 'neg': return -a[0]
        if self.op == 'pow2': return a[0] ** 2
        if self.op == 'pow3': return a[0] ** 3
        return FUNCTIONS[self.op](a[0])

def _convert(t, depth=0):
    if depth > MAX_DEPTH:
        raise ValueError('Expression is too deeply nested')
    if isinstance(t, ast.Constant) and type(t.value) in (int, float):
        value = float(t.value)
        if not math.isfinite(value) or abs(value) > 1e6:
            raise ValueError('Numeric constants must be finite and <= 1e6 in absolute value')
        return Node('number', value=value)
    if isinstance(t, ast.Name) and t.id in VARIABLES:
        return Node('variable', value=t.id)
    if isinstance(t, ast.UnaryOp) and isinstance(t.op, (ast.USub, ast.UAdd)):
        a = _convert(t.operand, depth + 1)
        if isinstance(t.op, ast.UAdd): return a
        node = Node('neg', (a,))
    elif isinstance(t, ast.BinOp):
        if isinstance(t.op, ast.Pow):
            if not isinstance(t.right, ast.Constant) or type(t.right.value) is not int or t.right.value not in (2, 3):
                raise ValueError('Only literal integer powers **2 and **3 are allowed')
            node = Node('pow' + str(t.right.value), (_convert(t.left, depth + 1),))
        else:
            op = {ast.Add: 'add', ast.Sub: 'sub', ast.Mult: 'mul', ast.Div: 'div'}.get(type(t.op))
            if op is None: raise ValueError('Unsupported operator')
            node = Node(op, (_convert(t.left, depth + 1), _convert(t.right, depth + 1)))
    elif isinstance(t, ast.Call) and isinstance(t.func, ast.Name) and t.func.id in FUNCTIONS and len(t.args) == 1 and not t.keywords:
        node = Node(t.func.id, (_convert(t.args[0], depth + 1),))
    else:
        raise ValueError('Only x1,x2,x3, numeric literals and declared operators are allowed')
    # A purely numeric subtree counts as one constant. Variable expressions
    # are never simplified, so the score is independent of a CAS version.
    if all(a.op == 'number' for a in node.args):
        with np.errstate(all='raise'):
            value = float(node.evaluate(np.zeros((1, 3))))
        if not math.isfinite(value) or abs(value) > 1e6:
            raise ValueError('Folded constant outside allowed range')
        return Node('number', value=value)
    return node

def parse_formula(expression, enforce_limit=True):
    if not isinstance(expression, str) or len(expression) > 5000:
        raise ValueError('Expression must be a string of at most 5000 characters')
    node = _convert(ast.parse(expression, mode='eval').body)
    if enforce_limit and node.complexity > MAX_COMPLEXITY:
        raise ValueError(f'Complexity {node.complexity} exceeds {MAX_COMPLEXITY}')
    return node

def predict(node, X):
    X = np.asarray(X, dtype=float)
    if X.ndim != 2 or X.shape[1] != 3 or not np.isfinite(X).all():
        raise ValueError('X must be a finite array of shape (n,3)')
    with np.errstate(divide='raise', invalid='raise', over='raise', under='ignore'):
        values = np.broadcast_to(np.asarray(node.evaluate(X), dtype=float), (len(X),)).copy()
    if not np.isfinite(values).all():
        raise ValueError('Nonfinite predictions')
    return values

def accuracy_score(error):
    if not np.isfinite(error) or error < 0:
        raise ValueError('Error must be finite and nonnegative')
    return float(np.clip(np.log(ERROR_BAD / max(error, ERROR_GOOD)) /
                         np.log(ERROR_BAD / ERROR_GOOD), 0, 1))

def score(error, complexity, bonus=0):
    if bonus not in (0, 10, 20):
        raise ValueError('Recovery bonus must be adjudicated as 0, 10 or 20')
    return 100 * accuracy_score(error) * math.exp(-complexity / COMPLEXITY_SCALE) + bonus

def combined_score(stage1, stage2=None):
    # Missing stage 2 means the first formula is retained.
    return 0.75 * stage1 + 0.25 * max(stage1, stage1 if stage2 is None else stage2)

def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--expression', required=True)
    ap.add_argument('--data', help='CSV with x1,x2,x3,y, for local RMSE only')
    ap.add_argument('--metadata', help='Public metadata.json with scale_y')
    args = ap.parse_args()
    node = parse_formula(args.expression)
    result = {'complexity': node.complexity, 'grammar_valid': True}
    if args.data:
        data = np.genfromtxt(args.data, delimiter=',', names=True)
        X = np.column_stack([data[k] for k in VARIABLES])
        yhat = predict(node, X)
        rmse = float(np.sqrt(np.mean((yhat - data['y'])**2)))
        result['rmse_noisy_data'] = rmse
        if args.metadata:
            scale = json.loads(Path(args.metadata).read_text())['scale_y']
            result['nrmse_noisy_data'] = rmse / scale
            result['proxy_points_not_hidden_score'] = score(rmse / scale, node.complexity)
    print(json.dumps(result, indent=2, ensure_ascii=False))

if __name__ == '__main__':
    main()
