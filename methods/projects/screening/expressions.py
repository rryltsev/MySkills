"""Restricted mathematical AST: no eval, imports, attributes or protected functions.

Complexity is a pilot convention, not an algebraic invariant. See PROTOCOL.md.
"""
import ast
import math
import numpy as np


def parse(expression):
    if len(expression) > 20000:
        raise ValueError("Expression too long")
    tree = ast.parse(expression, mode="eval").body
    if len(list(ast.walk(tree))) > 2000:
        raise ValueError("Expression too large")
    _walk(tree, np.ones((1, 2)))
    return tree


def _number(n):
    if isinstance(n, ast.Constant) and type(n.value) in (int, float):
        if not math.isfinite(n.value):
            raise ValueError("Nonfinite constant")
        return float(n.value)
    if isinstance(n, ast.UnaryOp) and isinstance(n.op, (ast.USub, ast.UAdd)):
        v = _number(n.operand)
        return -v if isinstance(n.op, ast.USub) else v
    raise ValueError("Not a numeric literal")


def _walk(n, x):
    if isinstance(n, ast.Constant):
        return _number(n), 1
    if isinstance(n, ast.Name) and n.id in ("x1", "x2"):
        return x[:, int(n.id[1]) - 1], 1
    if isinstance(n, ast.UnaryOp) and isinstance(n.op, (ast.USub, ast.UAdd)):
        try:
            return _number(n), 1
        except ValueError:
            v, c = _walk(n.operand, x)
            return (-v if isinstance(n.op, ast.USub) else v), c + 1
    if isinstance(n, ast.Call) and isinstance(n.func, ast.Name) and len(n.args) == 1 and not n.keywords:
        funcs = {"exp": np.exp, "log": np.log, "sqrt": np.sqrt}
        if n.func.id not in funcs:
            raise ValueError("Unsupported function")
        v, c = _walk(n.args[0], x)
        return funcs[n.func.id](v), c + 3
    if isinstance(n, ast.BinOp):
        a, ca = _walk(n.left, x)
        if isinstance(n.op, ast.Pow):
            p = _number(n.right)
            if p not in (2, 3):
                raise ValueError("Only fixed powers 2 and 3 are allowed")
            return a ** int(p), ca + 1
        b, cb = _walk(n.right, x)
        ops = {ast.Add: np.add, ast.Sub: np.subtract, ast.Mult: np.multiply, ast.Div: np.divide}
        if type(n.op) not in ops:
            raise ValueError("Unsupported operator")
        return ops[type(n.op)](a, b), ca + cb + 1
    raise ValueError("Unsupported syntax")


def evaluate(expression, x):
    with np.errstate(all="ignore"):
        value, c = _walk(parse(expression), np.asarray(x, dtype=float))
        y = np.broadcast_to(np.asarray(value, dtype=float), (len(x),)).copy()
    return y, c


def complexity(expression):
    with np.errstate(all="ignore"):
        return _walk(parse(expression), np.ones((1, 2)))[1]


def sympy_export(expr):
    """Normalize a PySR SymPy expression to the common grammar."""
    import sympy as s
    if expr.is_Number:
        return repr(float(expr))
    if expr.is_Symbol:
        if str(expr) not in ("x1", "x2"):
            raise ValueError("Unknown variable")
        return str(expr)
    if expr.is_Add or expr.is_Mul:
        op = "+" if expr.is_Add else "*"
        return "(" + op.join(sympy_export(a) for a in expr.args) + ")"
    if expr.is_Pow:
        a, b = expr.args
        base = sympy_export(a)
        if b == s.Rational(1, 2):
            return f"sqrt({base})"
        if b == s.Rational(-1, 2):
            return f"(1/sqrt({base}))"
        if b == -1:
            return f"(1/({base}))"
        if b in (2, 3):
            return f"(({base})**{int(b)})"
        if b in (-2, -3):
            return f"(1/(({base})**{int(-b)}))"
        # SymPy collapses repeated multiplication and nested square roots.
        # Re-expand dyadic rational powers using only the declared grammar.
        if b.is_Rational:
            numerator, denominator = int(b.p), int(b.q)
            if denominator > 0 and denominator & (denominator-1) == 0 and abs(numerator) <= 1000:
                rooted = base
                for _ in range(denominator.bit_length()-1):
                    rooted = f"sqrt({rooted})"
                n = abs(numerator)
                if n == 0:
                    powered = "1.0"
                elif n == 1:
                    powered = rooted
                elif n in (2,3):
                    powered = f"(({rooted})**{n})"
                else:
                    powered = "("+"*".join(f"({rooted})" for _ in range(n))+")"
                return f"(1/({powered}))" if numerator < 0 else powered
        raise ValueError("Export introduced unsupported power; retain raw formula for audit")
    if expr.func in (s.exp, s.log):
        return f"{expr.func.__name__}({sympy_export(expr.args[0])})"
    raise ValueError(f"Unsupported symbolic export: {expr.func}")
