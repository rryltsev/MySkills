"""Strain Hessian and Voigt-Reuss-Hill averaging from the CoCrNi QHA workflow."""
import numpy as np

def strain_matrix(eta):
 """Engineering Voigt Lagrangian strain -> symmetric deformation gradient."""
 e=np.array([[eta[0],eta[5]/2,eta[4]/2],[eta[5]/2,eta[1],eta[3]/2],[eta[4]/2,eta[3]/2,eta[2]]])
 w,v=np.linalg.eigh(np.eye(3)+2*e)
 if min(w)<=0: raise ValueError('Non-positive metric')
 return (v*np.sqrt(w))@v.T

def strained(a,eta):
 b=a.copy();b.set_cell(a.cell.array@strain_matrix(eta).T,scale_atoms=True);return b

def moduli(C):
 C=(np.array(C)+np.array(C).T)/2
 eig=np.linalg.eigvalsh(C)
 if min(eig)<=0: return dict(stable=False,eigenvalues_GPa=eig)
 S=np.linalg.inv(C);diag=np.trace(C[:3,:3]);off=C[0,1]+C[0,2]+C[1,2]
 Bv=(diag+2*off)/9;Gv=(diag-off+3*np.trace(C[3:,3:]))/15
 Br=1/(np.trace(S[:3,:3])+2*(S[0,1]+S[0,2]+S[1,2]))
 Gr=15/(4*np.trace(S[:3,:3])-4*(S[0,1]+S[0,2]+S[1,2])+3*np.trace(S[3:,3:]))
 B=(Bv+Br)/2;G=(Gv+Gr)/2
 return dict(stable=True,B_V_GPa=Bv,B_R_GPa=Br,G_V_GPa=Gv,G_R_GPa=Gr,B_H_GPa=B,G_H_GPa=G,E_H_GPa=9*B*G/(3*B+G),eigenvalues_GPa=eig)

def strain_states(h):
 yield 'zero',np.zeros(6)
 for i in range(6):
  for s in [-1,1]:
   eta=np.zeros(6);eta[i]=s*h;yield f'd{i}_{s}',eta
 for i in range(6):
  for j in range(i+1,6):
   for s in [-1,1]:
    for t in [-1,1]:
     eta=np.zeros(6);eta[i]=s*h;eta[j]=t*h;yield f'm{i}_{j}_{s}_{t}',eta

def hessian(values,h):
 nt=len(values['zero']);H=np.zeros((nt,6,6))
 for i in range(6): H[:,i,i]=(values[f'd{i}_1']-2*values['zero']+values[f'd{i}_-1'])/h**2
 for i in range(6):
  for j in range(i+1,6):
   x=sum(s*t*values[f'm{i}_{j}_{s}_{t}'] for s in [-1,1] for t in [-1,1])/(4*h*h);H[:,i,j]=x;H[:,j,i]=x
 return H
