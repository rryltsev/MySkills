import numpy as np
import pytest
from gfa.model_comparison import matched_errors,frozen_mode_curve

def test_sorted_vasp_atoms_and_elemental_offsets():
    # Input Zr,Cu,Ni is sorted Cu,Ni,Zr by the VASP writer.
    pred=dict(N=3,energy=-18.,forces=[[1,2,3],[4,5,6],[7,8,9]],stress=[0,0,0,0,0,0])
    ref=dict(N=3,energy_eV=-21.,input_order_to_poscar=[1,2,0],forces_eV_A=[[4,5,6],[7,8,9],[1,2,3]],stress_eV_A3=[0,0,0,0,0,0])
    force,stress,raw,aligned=matched_errors(pred,ref,['Zr','Cu','Ni'],dict(Zr=2,Cu=1,Ni=0))
    assert np.all(force==0) and np.all(stress==0)
    assert raw==1 and aligned==0

def test_missing_endpoint_does_not_fit_test_set_offset():
    pred=dict(N=1,energy=2.,forces=[[0,0,0]],stress=[0]*6)
    ref=dict(N=1,energy_eV=1.,input_order_to_poscar=[0],forces_eV_A=[[0,0,0]],stress_eV_A3=[0]*6)
    assert matched_errors(pred,ref,['Ni'],{})[-1] is None
    ref['input_order_to_poscar']=[1]
    with pytest.raises(ValueError):matched_errors(pred,ref,['Ni'],{})


def test_mode_curvature_sign_force_derivative_and_per_atom_normalization():
    # E(q)=E0 + k*q^2/2, where two atoms move oppositely along x.
    h=.02;k=-.8;direction=np.array([[1,0,0],[-1,0,0]])
    energies=[-10+k*q*q/2 for q in [-h,0,h]]
    forces=[-k*q*direction/2 for q in [-h,0,h]]
    curve=frozen_mode_curve(energies,forces,direction,h)
    assert curve['central_curvature_eV_atom_A2']==pytest.approx(k/2)
    assert curve['force_curvature_eV_atom_A2']==pytest.approx(k/2)
