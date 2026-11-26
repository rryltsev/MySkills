"""Three hand-checkable calculations, using only the Python standard library."""

from math import isclose
from statistics import mean


def main():
    # (parent configuration, partition, atom count, total DFT eV, total model eV)
    labels = [("solid-A", "train", 2, -6.1, -6.0),
              ("liquid-B", "train", 4, -9.0, -8.8),
              ("interface-C", "validation", 2, -4.3, -4.0)]
    shift = mean((dft - model) / n for _, split, n, dft, model in labels if split == "train")
    _, _, n, dft, model = labels[-1]
    error = (model + n * shift - dft) / n
    print(f"Training offset = {shift:+.3f} eV/atom. Signed validation error = {error:+.3f} eV/atom.")
    if not isclose(shift, -.05) or not isclose(error, .10):
        raise ValueError("Unexpected training offset or validation error")

    # One A center sees three A and one B on average. B occupies half of all sites.
    alpha_ab = 1 - (1 / 4) / .5
    print(f"P(B|A) = 0.25. Directed Warren-Cowley alpha_AB = {alpha_ab:+.2f}.")
    if not isclose(alpha_ab, .50):
        raise ValueError("Unexpected ordering parameter")

    # Same model/state/reference pressure for all three molar-atom enthalpies.
    x_b, h_alloy, h_a, h_b = .25, -2.30, -2., -2.8
    delta_h = h_alloy - ((1 - x_b) * h_a + x_b * h_b)
    print(f"Mixing enthalpy = {delta_h:+.3f} eV/atom = {delta_h*96.48533:+.2f} kJ/mol(atom).")
    if not isclose(delta_h, -.10):
        raise ValueError("Unexpected enthalpy")


if __name__ == "__main__":
    main()
