"""B2 geometric/chemical front diagnostics, independent of origin labels in the interface."""
import gzip
from pathlib import Path
import numpy as np
from ase import Atoms
from scipy.special import sph_harm_y
from scipy.ndimage import gaussian_filter1d
from .descriptors import neighbors

def read_frames(path,elements):
    path=Path(path);opener=gzip.open(path,'rt') if path.suffix=='.gz' else path.open();frames=[]
    with opener as f:
        while True:
            head=f.readline()
            if not head:break
            assert head.strip()=='ITEM: TIMESTEP';step=int(f.readline());assert f.readline().startswith('ITEM: NUMBER')
            n=int(f.readline());box_header=f.readline().strip()
            assert box_header in ['ITEM: BOX BOUNDS pp pp pp','ITEM: BOX BOUNDS xy xz yz pp pp pp']
            bounds=np.array([list(map(float,f.readline().split())) for _ in range(3)])
            if bounds.shape[1]==3:
                xy,xz,yz=bounds[:,2]
                xlo=bounds[0,0]-min(0,xy,xz,xy+xz);xhi=bounds[0,1]-max(0,xy,xz,xy+xz)
                ylo=bounds[1,0]-min(0,yz);yhi=bounds[1,1]-max(0,yz)
                origin=np.array([xlo,ylo,bounds[2,0]])
                cell=np.array([[xhi-xlo,0,0],[xy,yhi-ylo,0],[xz,yz,bounds[2,1]-bounds[2,0]]])
            else:
                origin=bounds[:,0];cell=np.diag(bounds[:,1]-bounds[:,0])
            columns=f.readline().split()[2:];values=np.fromstring(' '.join(f.readline() for _ in range(n)),sep=' ').reshape(n,len(columns))
            data={k:values[:,i] for i,k in enumerate(columns)};assert np.isfinite(values).all()
            assert np.array_equal(data['id'],np.arange(1,n+1));species=data['type'].astype(int)-1;assert set(species)<={0,1}
            xyz=np.c_[data['xu'],data['yu'],data['zu']]-origin
            a=Atoms(np.array(elements)[species],positions=xyz,cell=cell,pbc=True);a.wrap()
            frames.append(dict(step=step,atoms=a,species=species,data=data))
    return frames

def local_order(atoms,species,reciprocal_xy,cutoff=3.8):
    i,j,_,v=neighbors(atoms,cutoff);n=len(atoms);cn=np.bincount(i,minlength=n);assert min(cn)>0
    r=np.linalg.norm(v,axis=1);theta=np.arccos(np.clip(v[:,2]/r,-1,1));phi=np.arctan2(v[:,1],v[:,0]);qlm=np.zeros((n,13),complex)
    for k,m in enumerate(range(-6,7)):np.add.at(qlm[:,k],i,sph_harm_y(6,m,theta,phi))
    qlm/=cn[:,None];averaged=qlm.copy();np.add.at(averaged,i,qlm[j]);averaged/=(cn+1)[:,None]
    qbar=np.sqrt(4*np.pi/13*np.sum(abs(averaged)**2,axis=1))
    charge=1-2*np.asarray(species);coherence=np.tile(charge[:,None],(1,2)).astype(complex)
    np.add.at(coherence,i,charge[j,None]*np.exp(1j*(v@np.asarray(reciprocal_xy).T)))
    chemical=np.mean(abs(coherence)/(cn+1)[:,None],axis=1)
    return np.c_[qbar,chemical]

def fit_thresholds(positive,liquid,shuffled):
    thresholds={};separable=True
    for s in [0,1]:
        p=np.concatenate([x[0][x[1]==s] for x in positive]);l=np.concatenate([x[0][x[1]==s] for x in liquid]);z=np.concatenate([x[0][x[1]==s] for x in shuffled])
        pm=np.median(p,axis=0);lm=np.median(l,axis=0);zm=np.median(z,axis=0);negative=np.array([lm[0],max(lm[1],zm[1])])
        separable &= bool(np.all(pm>negative));thresholds[str(s)]=((pm+negative)/2).tolist()
    return dict(thresholds=thresholds,training_medians_separable=separable,rule='qbar6 > threshold AND local B2 chemical coherence > threshold; per central species')

def classify(values,species,model):
    thresholds=np.array([model['thresholds'][str(s)] for s in species]);return np.all(values>thresholds,axis=1)

def periodic_front(z,solid,Lz,bin_width=1.5,sigma=1.):
    bins=max(8,round(Lz/bin_width));edges=np.linspace(0,Lz,bins+1);counts=np.histogram(np.asarray(z)%Lz,edges)[0]
    hot=np.histogram(np.asarray(z)%Lz,edges,weights=np.asarray(solid,float))[0]
    density=gaussian_filter1d(counts.astype(float),sigma,mode='wrap');valid=bool(np.all(density>1e-12))
    fraction=np.divide(gaussian_filter1d(hot,sigma,mode='wrap'),density,out=np.zeros_like(density),where=density>1e-12)
    centers=(edges[:-1]+edges[1:])/2;crossings=[]
    for i in range(bins):
        j=(i+1)%bins
        if (fraction[i]>.5)==(fraction[j]>.5):continue
        x=centers[i]+(.5-fraction[i])/(fraction[j]-fraction[i])*(Lz/bins);crossings.append(float(x%Lz))
    crossings=sorted(crossings);widths=[]
    if len(crossings)==2:
        widths=[crossings[1]-crossings[0],Lz-(crossings[1]-crossings[0])]
    return dict(bin_centers_A=centers.tolist(),solid_fraction=fraction.tolist(),counts=counts.tolist(),crossings_A=crossings,
        exactly_two_fronts=len(crossings)==2 and valid,valid_density_profile=valid,domain_widths_A=widths,solid_equivalent_thickness_A=float(np.mean(fraction)*Lz))

def pressure_from_atomic_stress(frame):
    data=frame['data'];s=np.array([data[f'c_sat[{i}]'].sum() for i in range(1,10)])
    # LAMMPS centroid order: xx yy zz xy xz yz yx zx zy. Stress = -pressure*volume.
    return -np.array([s[0],s[1],s[2],(s[3]+s[6])/2,(s[4]+s[7])/2,(s[5]+s[8])/2])/frame['atoms'].get_volume()
