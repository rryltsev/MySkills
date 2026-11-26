import numpy as np
from ase import Atoms
from gfa.interface_screening import read_frames,local_order,periodic_front,pressure_from_atomic_stress


def test_triclinic_dump_origin_images_and_reciprocal_coherence(tmp_path):
    from ase.io import read
    # Nonzero box origin, both signs of tilt, and unwrapped periodic images.
    path=tmp_path/'tilted.dump'
    path.write_text('ITEM: TIMESTEP\n0\nITEM: NUMBER OF ATOMS\n2\n'
        'ITEM: BOX BOUNDS xy xz yz pp pp pp\n-3 11 2\n4 18 -1\n-2 28 3\n'
        'ITEM: ATOMS id type xu yu zu\n1 1 20 6 2\n2 2 5 16 24\n')
    actual=read_frames(path,['Cu','Zr'])[0]['atoms']
    expected=read(path,format='lammps-dump-text',specorder=['Cu','Zr'])
    np.testing.assert_allclose(actual.cell,expected.cell)
    expected.positions-=expected.get_celldisp().ravel()
    expected.wrap();np.testing.assert_allclose(actual.positions,expected.positions,atol=1e-12)
    a=3.3;atoms=Atoms('CuZr',scaled_positions=[[0,0,0],[.5,.5,.5]],cell=[a]*3,pbc=True).repeat((4,4,6))
    cell=atoms.cell.array.copy();cell[1,0]=.6;cell[2,:2]=[-.4,.5];atoms.set_cell(cell,scale_atoms=True)
    reciprocal=2*np.pi*4*np.linalg.inv(cell).T[:2]
    values=local_order(atoms,(atoms.numbers==40).astype(int),reciprocal)
    np.testing.assert_allclose(values[:,1],1,atol=1e-12)

def test_b2_chemical_coherence_rejects_species_shuffle_and_keeps_translation():
    a=3.3;atoms=Atoms('CuZr',scaled_positions=[[0,0,0],[.5,.5,.5]],cell=[a]*3,pbc=True).repeat((4,4,6))
    s=(atoms.numbers==40).astype(int);g=np.array([[2*np.pi/a,0,0],[0,2*np.pi/a,0]])
    x=local_order(atoms,s,g);np.testing.assert_allclose(x[:,1],1,atol=1e-12)
    atoms.positions+=[.37,1.26,2.43];atoms.wrap();np.testing.assert_allclose(local_order(atoms,s,g),x,atol=1e-12)
    shuffled=np.random.default_rng(72).permutation(s);y=local_order(atoms,shuffled,g)
    assert np.median(y[:,1])<.4;np.testing.assert_allclose(y[:,0],x[:,0])

def test_periodic_front_finds_two_interfaces_and_refuses_uniform_cell():
    z=np.linspace(0,60,6000,endpoint=False);r=periodic_front(z,(z<30),60)
    assert r['exactly_two_fronts'];np.testing.assert_allclose(sorted(r['domain_widths_A']),[30,30],atol=.1)
    assert not periodic_front(z,np.ones(len(z),bool),60)['exactly_two_fronts']
    shifted=(z+17)%60;r=periodic_front(shifted,z<30,60);assert r['exactly_two_fronts']
    np.testing.assert_allclose(sorted(r['domain_widths_A']),[30,30],atol=.1)

def test_centroid_stress_sign_and_offdiagonal_symmetrization():
    a=Atoms('CuZr',positions=[[0,0,0],[1,1,1]],cell=[10]*3,pbc=True)
    values=[1.,2,3,4,5,6,8,9,10];frame={'atoms':a,'data':{f'c_sat[{i+1}]':np.array([v,v]) for i,v in enumerate(values)}}
    np.testing.assert_allclose(pressure_from_atomic_stress(frame),-2*np.array([1,2,3,6,7,8])/1000)
