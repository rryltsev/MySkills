"""Numerical derivative checks, independent of MD integrators."""
import numpy as np

def derivatives(atoms, calculator, h=1e-5, strain=1e-5):
    atoms.calc = calculator
    energy = float(atoms.get_potential_energy())
    forces = atoms.get_forces()
    stress = atoms.get_stress(voigt=False)
    force_errors = []
    # Check chemically different centers and all Cartesian directions.
    for i in range(min(2,len(atoms))):
        for k in range(3):
            energies = []
            for sign in (-1,1):
                a = atoms.copy()
                a.positions[i,k] += sign*h
                a.calc = calculator
                energies.append(a.get_potential_energy())
            force_errors.append(abs(forces[i,k] + (energies[1]-energies[0])/(2*h)))
    stress_errors = []
    for i,j in ((0,0),(1,1),(2,2),(1,2),(0,2),(0,1)):
        energies = []
        for sign in (-1,1):
            f = np.eye(3)
            if i == j:
                f[i,j] += sign*strain
            else:
                f[i,j] += sign*strain/2
                f[j,i] += sign*strain/2
            a = atoms.copy()
            a.set_cell(atoms.cell.array @ f.T, scale_atoms=True)
            a.calc = calculator
            energies.append(a.get_potential_energy())
        numerical = (energies[1]-energies[0])/(2*strain*atoms.get_volume())
        stress_errors.append(abs(stress[i,j]-numerical))
    return {"energy_eV_atom": energy/len(atoms),"max_force_derivative_error_eV_A":float(max(force_errors)),
            "max_stress_derivative_error_eV_A3":float(max(stress_errors)),
            "net_force_eV_A":float(np.linalg.norm(forces.sum(axis=0))),
            "finite":bool(np.isfinite(energy) and np.isfinite(forces).all() and np.isfinite(stress).all())}
