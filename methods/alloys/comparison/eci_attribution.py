"""Attribute differences between two code-compatible Ti-Al MFA Hamiltonians."""
import argparse
import csv
import sys
from pathlib import Path

import numpy as np
from numpy.polynomial.legendre import leggauss
from scipy.optimize import least_squares

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/"thermo"))
from mfa import Isotherm, KB, ORBIT_NAMES, lower_hull_edges, sublattice_weights
from hcp_orbits import MULTIPLICITIES, VREF

NAMES=ORBIT_NAMES[2:]
DEGREES=np.array([2]*7+[3]*11)


def derivative(poly,du,dv):
    result=np.zeros_like(poly)
    result[:,:-1,:]+=du*poly[:,1:,:]*np.arange(1,poly.shape[1])[None,:,None]
    result[:,:,:-1]+=dv*poly[:,:,1:]*np.arange(1,poly.shape[2])[None,None,:]
    return result


class Tangents:
    def __init__(self, temperature):
        self.temperature = temperature
        weights=sublattice_weights()
        self.weights=weights
        phi=weights[2:]
        dc=derivative(phi,2,2)
        dm=derivative(phi,1.5,-.5)
        self.basis=np.stack([phi,dc,dm,derivative(dc,2,2),derivative(dc,1.5,-.5),derivative(dm,1.5,-.5)])

    def state(self,j,c,m):
        q,r=c+.75*m,c-.25*m
        if not (0<q<1 and 0<r<1):
            raise ValueError("Invalid sublattice filling")
        u,v=2*q-1,2*r-1
        polys=np.einsum("kaij,i,j->ka",self.basis,u**np.arange(4),v**np.arange(4))
        h=lambda p:p*np.log(p)+(1-p)*np.log1p(-p)
        lq,lr=np.log(q/(1-q)),np.log(r/(1-r))
        sq,sr=1/(q*(1-q)),1/(r*(1-r))
        ent=KB*self.temperature/4*np.array([h(q)+3*h(r),lq+3*lr,.75*(lq-lr),sq+3*sr,
                            .75*(sq-sr),.5625*sq+.1875*sr])
        f=polys@j+ent
        return f,polys

    def solve(self,j,initial,check=True):
        # Variables (c_alpha,q_alpha2,r_alpha2) keep both probabilities bounded.
        def equations(x):
            a,q,r=x
            b,m=(q+3*r)/4,q-r
            fa,_=self.state(j,a,0); fb,_=self.state(j,b,m)
            return np.array([fa[1]-fb[1], (fa[1]*(b-a)-(fb[0]-fa[0]))/.12, fb[2]])
        refine=lambda x:least_squares(equations,x,bounds=(1e-11,1-1e-11),ftol=1e-14,xtol=1e-14,gtol=1e-14)
        def distinct(sol):
            a,q,r=sol.x
            b=(q+3*r)/4
            return sol.success and np.max(abs(sol.fun))<1e-9 and a+.005<b<.35 and q-r>.05
        sol=refine(initial)
        grid=np.linspace(0,.35,701)
        feasible=Isotherm(self.weights,np.r_[0.,0.,j],self.temperature)
        energy=None
        if not distinct(sol):
            energy,q_grid=feasible.equilibrium(grid)
            gaps=[(l,r) for l,r in lower_hull_edges(grid,energy) if grid[r]-grid[l]>.005]
            if len(gaps)!=1 or gaps[0][0]==0 or gaps[0][1]==len(grid)-1:
                raise ValueError(f"No unique interior coexistence on the global hull: {gaps}")
            left,right=gaps[0]
            q=q_grid[right]
            sol=refine(np.clip([grid[left],q,(4*grid[right]-q)/3],1e-9,1-1e-9))
        a,q,r=sol.x; b,m=(q+3*r)/4,q-r
        if (not sol.success or np.max(np.abs(equations(sol.x)))>1e-9 or b-a<.005 or m<.05 or b>.35):
            raise ArithmeticError(f"Lost distinct common tangent: {sol.x}, residuals {sol.fun}")
        fa,pa=self.state(j,a,0); fb,pb=self.state(j,b,m)
        states=[(a,0,fa,pa),(b,m,fb,pb)]
        width=b-a
        # d(mu_coex)/dJ from differentiating the equal grand potentials.
        dmu=(pb[0]-pa[0])/width
        random=lambda c: MULTIPLICITIES[2:]*(2*c-1)**DEGREES
        random_prime=lambda c: 2*MULTIPLICITIES[2:]*DEGREES*(2*c-1)**(DEGREES-1)
        random_slope=(random(b)-random(a))/width
        deriv=[]; background=[]; curv=[]
        for c,order,f,p in states:
            k=f[3]-f[4]**2/f[5]
            if k<=0 or f[5]<=0:
                raise ValueError("Unstable chemical or sublattice curvature at a contact")
            curv.append(k)
            dm_dc=-f[4]/f[5]
            deriv.append((dmu-p[1]-p[2]*dm_dc)/k)
            background.append((random_slope-random_prime(c))/k)
        if check:
            if energy is None:
                energy,_=feasible.equilibrium(grid)
            margin=np.min(energy-(fa[0]+fa[1]*(grid-a)))
            endpoint_energy,_=feasible.equilibrium(np.array([a,b]))
            if margin< -2e-8 or max(abs(endpoint_energy-[fa[0],fb[0]]))>2e-8:
                raise ValueError(f"Failed global supporting-line check: margin {margin:.3g}")
        return {"x":sol.x,"c":np.array([a,b]),"q":q,"r":r,"m":m,"derivatives":np.array(deriv),
                "background":np.array(background),"curvature":curv,
                "order_phi":pb[0]-random(b),"tangent_mu_gauge":float(fa[1])}


def read_grid(path, temperature, static_only):
    with np.load(path, allow_pickle=False) as data:
        if tuple(data["orbit_names"]) != ORBIT_NAMES:
            raise ValueError("Unexpected orbit names or order")
        iv, = np.flatnonzero(np.isclose(data["volume_A3_atom"], VREF, rtol=1e-7))
        values = data["J_static"][iv, 2:].copy()
        if not static_only:
            it, = np.flatnonzero(np.isclose(data["temperature_K"], temperature, rtol=0, atol=1e-7))
            values += data["J_vib"][iv, it, 2:]
    return values


def contributions(model, start, end, order):
    delta = end - start
    nodes, weights = leggauss(order)
    integrals = np.zeros((2, len(delta)))
    initial = [.08, .76, .025]
    for node, weight in zip(nodes, weights):
        state = model.solve(start + .5 * (1 + node) * delta, initial)
        integrals += .5 * weight * state["derivatives"]
        initial = state["x"]
    return 100 * integrals * delta


if __name__ == "__main__":
    cli = argparse.ArgumentParser(description=__doc__)
    cli.add_argument("start", type=Path)
    cli.add_argument("end", type=Path)
    cli.add_argument("--temperature", type=float, required=True)
    cli.add_argument("--static-only", action="store_true")
    cli.add_argument("--out", type=Path, required=True, help="new CSV")
    args = cli.parse_args()
    if not np.isfinite(args.temperature) or args.temperature <= 0:
        cli.error("temperature must be positive and finite")
    j0, j1 = [read_grid(path, args.temperature, args.static_only) for path in (args.start, args.end)]
    model = Tangents(args.temperature)
    s0 = model.solve(j0, [.08, .76, .025])
    s1 = model.solve(j1, s0["x"])
    expected = 100 * (s1["c"] - s0["c"])
    result = contributions(model, j0, j1, 16)
    np.testing.assert_allclose(result, contributions(model, j0, j1, 8), rtol=0, atol=1e-6)
    np.testing.assert_allclose(result.sum(axis=1), expected, rtol=0, atol=1e-6)
    with args.out.open("x", encoding="utf8", newline="") as output:
        writer = csv.writer(output)
        writer.writerow(["orbit", "shift_left_at_pct", "shift_right_at_pct"])
        writer.writerows(zip(NAMES, *result))
    print("Endpoint shifts in percentage points:", expected)
