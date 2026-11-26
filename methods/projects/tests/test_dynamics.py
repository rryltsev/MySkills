import numpy as np
import pytest
from ase import Atoms
from gfa.dynamics import self_isf,partial_rdf

def test_isf_removes_rigid_translation_but_keeps_relative_motion():
    xyz=np.zeros((3,2,3));xyz[1]=[5,0,0];xyz[2]=[[6,0,0],[4,0,0]]
    f,msd=self_isf(xyz,np.pi)
    assert np.allclose(f,[1,1,0],atol=1e-12)
    assert np.allclose(msd,[0,0,1])

def test_partial_rdf_cross_pair_reciprocity():
    a=Atoms("Cu2Zr2",positions=[[1,1,1],[3,1,1],[2,2,1],[6,6,6]],cell=[12]*3,pbc=True)
    _,g=partial_rdf(a,["Cu","Zr"],rmax=5)
    assert np.allclose(g["Cu-Zr"],g["Zr-Cu"])
    with pytest.raises(ValueError):partial_rdf(a,["Cu","Zr"],rmax=6)
