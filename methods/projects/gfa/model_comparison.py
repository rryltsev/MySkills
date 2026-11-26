"""Explicit atom-order and reference-energy conventions for MLIP/DFT comparisons."""
import numpy as np

EV_A3_TO_GPA=160.21766208


def frozen_mode_curve(energies, forces, direction, step):
    """Energy curvature and its independent force derivative for x(q)=x0+q*u.

    Energies are total eV and forces are in the same atom order as direction.
    The three inputs correspond to q=-step,0,+step; output is per atom.
    """
    energies=np.asarray(energies,dtype=float)
    forces=np.asarray(forces,dtype=float)
    direction=np.asarray(direction,dtype=float)
    if energies.shape!=(3,) or direction.ndim!=2 or direction.shape[1]!=3 or forces.shape!=(3,*direction.shape) or step<=0:
        raise ValueError('Invalid frozen-mode inputs')
    n=len(direction)
    return dict(amplitudes_A=[-step,0.,step],delta_energy_eV_atom=((energies-energies[1])/n).tolist(),
                central_curvature_eV_atom_A2=float((energies[0]+energies[2]-2*energies[1])/(n*step**2)),
                force_curvature_eV_atom_A2=float(-np.sum((forces[2]-forces[0])*direction)/(2*step*n)),
                reference_projected_force_eV_atom_A=float(np.sum(forces[1]*direction)/n),
                maximum_reference_force_eV_A=float(np.linalg.norm(forces[1],axis=1).max()))

def matched_errors(prediction, reference, symbols, offsets):
    n=len(symbols);order=np.asarray(reference['input_order_to_poscar'],dtype=int)
    if sorted(order.tolist())!=list(range(n)):raise ValueError('Invalid DFT atom permutation')
    if prediction['N']!=n or reference['N']!=n:raise ValueError('Different atom counts')
    force=np.asarray(prediction['forces'],dtype=float)[order]-np.asarray(reference['forces_eV_A'])
    stress=(np.asarray(prediction['stress'],dtype=float)-np.asarray(reference['stress_eV_A3']))*EV_A3_TO_GPA
    if force.shape!=(n,3) or stress.shape!=(6,):raise ValueError('Unexpected force/stress shape')
    energy=(prediction['energy']-reference['energy_eV'])/n
    aligned=energy-np.mean([offsets[s] for s in symbols]) if all(s in offsets for s in symbols) else None
    if not np.isfinite(force).all() or not np.isfinite(stress).all() or not np.isfinite(energy):raise ValueError('Non-finite prediction/label')
    return force,stress,float(energy),None if aligned is None else float(aligned)
