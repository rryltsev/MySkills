"""Run the HCP, free-energy and phase-equilibrium examples on synthetic inputs."""
import argparse
import math
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent
for module_dir in ("atomistic", "thermo", "alloy_ti"):
    sys.path.insert(0, str(ROOT / module_dir))

from decorations import decoration_orbits
from fit_eci import fit_volume
from hcp_orbits import MULTIPLICITIES, ORBIT_NAMES, VREF, build_orbits, features
from label_cell import H_THZ, KB, harmonic_labels
from mfa import Isotherm, entropy_term, sublattice_weights
from pilot import estimate
from vcsgc import integrate


def verify_cluster_fit(rng):
    orbits = build_orbits((4, 4, 4))
    n = len(orbits[1])
    np.testing.assert_allclose(features(np.ones(n), orbits), MULTIPLICITIES)
    features_by_cell = [
        features(np.where(rng.random(n) < fraction, 1, -1), orbits)
        for fraction in np.linspace(.03, .95, 80)
    ]
    matrix = np.asarray(features_by_cell)
    known = rng.uniform(-.02, .02, len(ORBIT_NAMES))
    energy = (matrix @ known)[:, None]
    train = np.arange(len(matrix)) % 4 != 0
    fitted, _, scores = fit_volume(matrix, energy, train)
    np.testing.assert_allclose(fitted[:, 0], known, atol=1e-11)
    print(f"CE: rank {np.linalg.matrix_rank(matrix[train])}, held-out RMSE "
          f"{scores['control_rmse'][0]:.2g} eV/atom ({scores['control']} cells)")


def verify_phonons(temperature):
    # One Gamma point in a two-atom cell: three translations, three optical modes.
    frequencies = np.array([[0., 0., 0., 3., 4., 5.]])
    q, weights = np.zeros((1, 3)), np.ones(1)
    result, _ = harmonic_labels(frequencies, q, weights, [temperature], 2)
    kt = KB * temperature
    epsilon = H_THZ * frequencies[0, 3:]
    direct = kt * np.log(2 * np.sinh(epsilon / (2 * kt))).sum() / 2
    np.testing.assert_allclose(result, [direct], atol=1e-12)
    frequencies[0, -1] = -.3
    unstable, minimum = harmonic_labels(frequencies, q, weights, [temperature], 2)
    if unstable is not None or minimum >= 0:
        raise AssertionError("Expected the unstable mesh to be rejected")
    print(f"Quantum HA: Fvib {direct:.8f} eV/atom, unstable internal mode detected")


def verify_isotherms(temperature):
    model = Isotherm(sublattice_weights(), np.zeros(len(ORBIT_NAMES)), temperature)
    c = np.array([.08, .15, .25, .30])
    f, q = model.equilibrium(c)
    np.testing.assert_allclose(q, c, atol=1e-7)
    np.testing.assert_allclose(f, KB * temperature * entropy_term(c), atol=1e-12)
    # Exact polynomial with equal-depth minima at c=.1 and c=.3.
    c = np.linspace(.05, .35, 61)
    mu = 2 * (c - .1) * (c - .3) * (2*c - .4)
    _, _, contacts = integrate(c, mu, min_span=.05)
    if len(contacts) != 1 or contacts[0]["window_edge"]:
        raise AssertionError("Expected one internal tangent")
    ends = [contacts[0][name] for name in ("alpha_at_pct", "alpha2_at_pct")]
    np.testing.assert_allclose(ends, [10, 30], atol=.03)
    print(f"Ideal MFA: q=c. Integrated tangent: {ends[0]:.3f}, {ends[1]:.3f} at.%")


def verify_atomistic_reference(rng, temperature):
    groups = decoration_orbits()
    counts = {name: len(group) for name, group in groups.items()}
    if sum(counts.values()) != math.comb(8, 2):
        raise AssertionError("Incorrect number of decorations")
    # Harmonic reference with k=1, target with k=1.5 (eV/angstrom^2).
    kt = KB * temperature
    coordinates = rng.normal(0, np.sqrt(kt), 30000)
    delta_u = .25 * coordinates**2
    result = estimate(delta_u, np.zeros_like(delta_u), kt)
    expected = .5 * kt * np.log(1.5)
    np.testing.assert_allclose([result["ti_eV"], result["fep_eV"]], expected, atol=.0006)
    print(f"Decoration multiplicities: {counts}. Oscillator TI: "
          f"{result['ti_eV']:.6f} eV; exact {expected:.6f} eV")


if __name__ == "__main__":
    cli = argparse.ArgumentParser(description=__doc__)
    cli.add_argument("--example-dir", type=Path, help="write ideal-alloy inputs to a fresh directory")
    options = cli.parse_args()
    rng = np.random.default_rng(101)
    verify_cluster_fit(rng)
    verify_phonons(800.)
    verify_isotherms(800.)
    verify_atomistic_reference(rng, 800.)
    if options.example_dir:
        options.example_dir.mkdir(parents=True, exist_ok=False)
        zeros = np.zeros((1, len(ORBIT_NAMES)))
        np.savez(options.example_dir / "grid.npz", orbit_names=ORBIT_NAMES,
                 J_static=zeros, J_vib=zeros[:, None, :], volume_A3_atom=[VREF],
                 temperature_K=[800.])
        np.savetxt(options.example_dir / "coeff.txt", np.column_stack(([VREF], zeros)))
