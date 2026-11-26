"""Chemical neighbor descriptors with explicit periodic image identities.

Edges are directed (i,j,S). Their inverse is (j,i,-S), including self images.
No minimum-image shortcut: small cells can have multiple images of one atom.
Cutoffs must be symmetric, later determined from partial RDF minima.
"""
import numpy as np
from ase.neighborlist import neighbor_list
from scipy.special import sph_harm_y

def neighbors(atoms, cutoff):
    if isinstance(cutoff, dict):
        for (a, b), value in cutoff.items():
            if (b, a) in cutoff and cutoff[b, a] != value:
                raise ValueError("Pair cutoffs must be reciprocal")
    i, j, shifts, vectors = neighbor_list("ijSD", atoms, cutoff, self_interaction=False)
    edges = {(int(a), int(b), *map(int, s)) for a, b, s in zip(i, j, shifts)}
    if len(edges) != len(i):
        raise ValueError("Duplicate periodic edges")
    for a, b, x, y, z in edges:
        if (b, a, -x, -y, -z) not in edges:
            raise ValueError("Nonreciprocal neighborhood")
    if np.any(np.linalg.norm(vectors, axis=1) < 1e-10):
        raise ValueError("Overlapping atoms")
    return i, j, shifts, vectors

def local_descriptors(atoms, cutoff, elements):
    i, j, shifts, vectors = neighbors(atoms, cutoff)
    symbols = np.array(atoms.get_chemical_symbols())
    if not set(symbols) <= set(elements):
        raise ValueError("Incomplete chemical map")
    coordination = np.bincount(i, minlength=len(atoms))
    counts = np.column_stack([np.bincount(i[symbols[j] == s], minlength=len(atoms)) for s in elements])
    result = {"center_species": symbols, "coordination": coordination, "neighbor_counts": counts}
    radius = np.linalg.norm(vectors, axis=1)
    theta = np.arccos(np.clip(vectors[:, 2]/radius, -1, 1))
    phi = np.arctan2(vectors[:, 1], vectors[:, 0])
    # q_l for all bonds and each neighbor species; absent channels are NaN, not artificial zero order.
    for ell in (4, 6):
        for channel in (None, *elements):
            select = np.ones(len(i), bool) if channel is None else symbols[j] == channel
            n = np.bincount(i[select], minlength=len(atoms))
            qlm = np.zeros((len(atoms), 2*ell+1), complex)
            for k, m in enumerate(range(-ell, ell+1)):
                np.add.at(qlm[:, k], i[select], sph_harm_y(ell, m, theta[select], phi[select]))
            qlm /= np.maximum(n, 1)[:, None]
            q = np.sqrt(4*np.pi/(2*ell+1)*np.sum(abs(qlm)**2, axis=1))
            q[n == 0] = np.nan
            result[f"q{ell}_{channel or 'all'}"] = q
    return result

def shared_shells(atoms, cutoff):
    """Shared neighbors for each canonical bond, with both shells in i's image frame."""
    i, j, shifts, _ = neighbors(atoms, cutoff)
    shells = [set() for _ in atoms]
    for a, b, s in zip(i, j, shifts):
        shells[a].add((int(b), *map(int, s)))
    bonds, counts = [], []
    for a, b, s in zip(i, j, shifts):
        key = (int(a), int(b), *map(int, s))
        inverse = (int(b), int(a), *map(int, -s))
        if key > inverse:
            continue
        translated = {(k, x+int(s[0]), y+int(s[1]), z+int(s[2])) for k, x, y, z in shells[b]}
        bonds.append(key)
        counts.append(len(shells[a] & translated))
    return np.asarray(bonds, dtype=int), np.asarray(counts, dtype=int)

def motif_components(atoms, cutoff, mask):
    """Components in the periodic quotient graph; does not infer percolation/winding."""
    mask = np.asarray(mask, dtype=bool)
    if mask.shape != (len(atoms),):
        raise ValueError("Wrong motif mask")
    i, j, _, _ = neighbors(atoms, cutoff)
    parent = np.arange(len(atoms))
    def root(a):
        while parent[a] != a:
            parent[a] = parent[parent[a]]
            a = parent[a]
        return a
    for a, b in zip(i, j):
        if mask[a] and mask[b]:
            parent[root(b)] = root(a)
    groups = {}
    for a in np.flatnonzero(mask):
        groups.setdefault(root(a), []).append(int(a))
    return sorted(groups.values(), key=len, reverse=True)
