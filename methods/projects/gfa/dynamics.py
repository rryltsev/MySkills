"""Trajectory diagnostics; callers must provide physical MD, unwrapped positions and fixed volume."""
import numpy as np
from .descriptors import neighbors

def self_isf(positions_unwrapped, wavevector_magnitude, masses=None):
    """Single-origin orientational average Fs(k,t)=<sinc(k*dr)>; not an uncertainty estimate."""
    xyz=np.asarray(positions_unwrapped,dtype=float)
    if xyz.ndim!=3 or xyz.shape[2]!=3 or wavevector_magnitude<=0:
        raise ValueError("Expected (frames,atoms,3) and positive k")
    displacement=xyz-xyz[0]
    drift=np.average(displacement,axis=1,weights=masses)
    displacement-=drift[:,None,:]
    radius=np.linalg.norm(displacement,axis=2)
    return np.sinc(wavevector_magnitude*radius/np.pi).mean(axis=1),np.mean(radius**2,axis=1)

def partial_rdf(atoms,elements,rmax=5.0,bins=100):
    """Directed pair counts; canonical finite-N normalization, homogeneous isotropic bulk only."""
    if not np.all(atoms.pbc):raise ValueError("Fully periodic bulk required")
    heights=1/np.linalg.norm(np.linalg.inv(atoms.cell.array),axis=0)
    if rmax>=.5*min(heights):raise ValueError("rmax must be below half the smallest cell height")
    edges=np.linspace(0,rmax,bins+1);shell=4*np.pi/3*np.diff(edges**3)
    i,j,_,v=neighbors(atoms,rmax)
    symbols=np.asarray(atoms.get_chemical_symbols());r=np.linalg.norm(v,axis=1)
    curves={}
    for a in elements:
        for b in elements:
            na=np.count_nonzero(symbols==a);nb=np.count_nonzero(symbols==b)
            pairs=na*(nb-int(a==b))
            if pairs==0:continue
            counts=np.histogram(r[(symbols[i]==a)&(symbols[j]==b)],edges)[0]
            curves[f"{a}-{b}"]=counts*atoms.get_volume()/pairs/shell
    return (edges[:-1]+edges[1:])/2,curves
