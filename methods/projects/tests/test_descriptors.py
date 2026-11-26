import itertools
import numpy as np
from ase import Atoms
from ase.build import bulk
import pytest
from gfa.descriptors import neighbors, local_descriptors, shared_shells, motif_components

def test_one_atom_cell_keeps_six_distinct_images():
    a = Atoms("Cu", positions=[[0, 0, 0]], cell=[3, 3, 3], pbc=True)
    i, j, s, v = neighbors(a, 3.01)
    assert len(i) == 6 and np.all(i == j)
    assert len(set(map(tuple, s))) == 6
    assert np.allclose(np.linalg.norm(v, axis=1), 3)

def test_triclinic_against_explicit_image_enumeration():
    a = Atoms("CuZr", scaled_positions=[[.03,.12,.14],[.95,.85,.84]], cell=[[5,0,0],[1.2,4.5,0],[.8,.9,4.7]], pbc=True)
    cutoff = 3.1
    i, j, s, _ = neighbors(a, cutoff)
    actual = {(int(x), int(y), *map(int,z)) for x,y,z in zip(i,j,s)}
    expected = set()
    for x,y,z in itertools.product(range(2),range(2),itertools.product(range(-2,3),repeat=3)):
        d = a.positions[y] + np.array(z) @ a.cell.array - a.positions[x]
        if 1e-10 < np.linalg.norm(d) < cutoff:
            expected.add((x,y,*z))
    assert actual == expected

def test_fcc_order_and_shared_neighbors():
    a = bulk("Cu", "fcc", a=3.6, cubic=True).repeat((2,2,2))
    d = local_descriptors(a, 2.7, ["Cu", "Zr"])
    assert np.all(d["coordination"] == 12)
    assert np.allclose(d["q4_all"], .19094065, atol=1e-7)
    assert np.allclose(d["q6_all"], .57452426, atol=1e-7)
    assert np.isnan(d["q4_Zr"]).all()
    _, shared = shared_shells(a, 2.7)
    assert np.all(shared == 4)

def test_wrapping_rotation_and_chemical_counts():
    a = Atoms("CuZr", scaled_positions=[[0,0,0],[.5,.5,.5]], cell=[3.3]*3, pbc=True).repeat((2,2,2))
    d = local_descriptors(a, 3.0, ["Cu", "Zr"])
    assert np.all(d["coordination"] == 8)
    assert np.all(d["neighbor_counts"][::2, 1] == 8)
    original_shared = sorted(shared_shells(a, 3.0)[1])
    a.positions[1] += a.cell[0] - 2*a.cell[1]
    a.rotate(29, [1,2,3], rotate_cell=True)
    e = local_descriptors(a, 3.0, ["Cu", "Zr"])
    assert np.allclose(d["q6_all"], e["q6_all"])
    assert original_shared == sorted(shared_shells(a, 3.0)[1])
    assert len(motif_components(a, 3.0, np.ones(len(a),bool))[0]) == len(a)
    assert motif_components(a, 3.0, np.zeros(len(a),bool)) == []

def test_asymmetric_cutoffs_rejected():
    a = Atoms("CuZr", positions=[[0,0,0],[2,0,0]],cell=[10]*3,pbc=True)
    with pytest.raises(ValueError):
        neighbors(a, {("Cu","Zr"):3, ("Zr","Cu"):2})
