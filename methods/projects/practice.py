"""Small deterministic examples using the research code. Run with --help."""
import argparse
import math

import numpy as np


def motifs():
    from gfa.motif_lifetimes import lifetime

    # An episode at the left edge, a complete episode and an unfinished one.
    masks = np.array([[1, 0, 0], [1, 1, 0], [0, 1, 1], [0, 0, 1]], dtype=bool)
    result = lifetime(masks, dt=0.25, tau=2.0)
    assert result["left_censored_excluded"] == 1
    assert result["completed_episodes"] == result["right_censored"] == 1
    print("completed =", result["completed_episodes"], "right-censored =", result["right_censored"])
    print("survival =", result["KM_survival"], "RMST bounds, ps =", result["RMST_bounds_2ps"])


def viscosity():
    from oxide.composition_curve import OXIDES, fit, predict_curve
    from oxide.oxide_basis import MOLAR_MASS
    from oxide.study_weights import weights

    # One mole each, supplied as gram amounts, then expressed in the model's order.
    mass = {"sio2": MOLAR_MASS["sio2"], "cao": MOLAR_MASS["cao"]}
    amounts = np.array([mass.get(name, 0) / MOLAR_MASS[name] for name in OXIDES])
    molar = amounts / amounts.sum()
    np.testing.assert_allclose(molar[molar > 0], [0.5, 0.5])
    # A synthetic log10-viscosity curve shared by both compositions.
    train_t = np.tile([1450, 1550, 1650, 1750], 2)
    silica = np.eye(len(OXIDES))[OXIDES.index("sio2")]
    x = np.repeat([molar, silica], 4, axis=0)
    response = -4.7 + 8500.0 / train_t
    model = fit(x, np.zeros(8), train_t, response, length=0.3, count=2, order=2, ridge=1e-6)
    test_t = np.array([1500.0, 1700.0])
    prediction = predict_curve(model, np.tile(molar, (2, 1)), np.zeros(2), test_t)
    reference = -4.7 + 8500.0 / test_t
    np.testing.assert_allclose(prediction, reference, atol=0.005)
    print("mole fractions SiO2, CaO = 0.5, 0.5")
    print("log10(eta / Pa s) at 1500, 1700 K =", prediction)
    rows = [{"study": "A", "sample": "one"}] * 3 + [{"study": "B", "sample": "two"}]
    np.testing.assert_allclose(weights(rows), [1/6, 1/6, 1/6, 1/2])
    print("weights by source =", weights(rows))


def formulas():
    from io import BytesIO
    from fractions import Fraction
    from screening.domain import interval_check
    from screening.nn_replay import nn_predict
    from screening.selection import one_se
    from symbolic.formula_tools import parse_formula, predict
    from symbolic.native_tree import op_expression, op_replay, scaled_error

    node = parse_formula("x1**2 + x2")
    values = predict(node, np.array([[2.0, 3.0, 0.0], [3.0, -1.0, 0.0]]))
    np.testing.assert_array_equal(values, [7.0, 8.0])
    bounds = [[-1.0, 1.0], [-1.0, 1.0]]
    assert interval_check("1/(1+x1**2)", bounds)["status"] == "passed"
    assert interval_check("1/x1", bounds, max_depth=4)["status"] == "unresolved"
    candidates = [
        {"candidate_id": "long", "selection_complexity": 10},
        {"candidate_id": "short", "selection_complexity": 3},
    ]
    predictions = np.array([[0.0, 0.0, 0.0, 1.0], [0.6, 0.6, 0.6, 0.6]])
    index, result = one_se(candidates, predictions, np.zeros(4), sy=1.0)
    assert index == 1
    print("predictions =", values, "complexity =", node.complexity)
    print("validation reference =", result["reference_candidate_id"], "selected =", result["selected_candidate_id"])
    records = [
        {"type": "variable", "value": 0.4, "variable": "x1", "arity": 0, "length": 0},
        {"type": "variable", "value": 0.4, "variable": "x1", "arity": 0, "length": 0},
        {"type": "add", "value": 1.0, "arity": 2, "length": 2},
    ]
    x = np.array([[1.0, 0.0, 0.0], [-3.0, 0.0, 0.0]])
    replay = op_replay(records, x)
    exported = op_expression(records)
    np.testing.assert_array_equal(replay, predict(parse_formula(exported), x))
    approx = Fraction(0.8).limit_denominator(20)
    assert approx != Fraction.from_float(0.8)
    print("native export =", exported, "scaled Float32 discrepancy =", scaled_error(replay, op_replay(records, x, np.float32)))
    # Replay an independently frozen one-layer neural network in memory.
    frozen = BytesIO()
    np.savez(frozen, W0=np.ones((2, 1)), W1=np.ones((1, 1)), b0=np.zeros(1), b1=np.zeros(1),
             x_mean=np.zeros(2), x_scale=np.ones(2), y_mean=np.zeros(1), y_scale=np.ones(1))
    frozen.seek(0)
    np.testing.assert_array_equal(nn_predict(frozen, np.zeros((1, 2))), [0.5])


def elasticity():
    from elastic.qha_core import hessian, moduli, strain_states

    # Isotropic material with B=100 GPa and G=60 GPa, using engineering shear.
    expected = np.diag([180.0, 180.0, 180.0, 60.0, 60.0, 60.0])
    for i, j in [(0, 1), (0, 2), (1, 2)]:
        expected[i, j] = expected[j, i] = 60.0
    step = 0.005
    # F/V in units of GPa on an analytic surface.
    values = {name: np.array([0.5 * strain @ expected @ strain]) for name, strain in strain_states(step)}
    tensor = hessian(values, step)[0]
    np.testing.assert_allclose(tensor, expected, atol=1e-10)
    result = moduli(tensor)
    assert result["stable"] and math.isclose(result["B_H_GPa"], 100.0)
    print("strain states =", len(values), "B_H, G_H in GPa =", result["B_H_GPa"], result["G_H_GPa"])


def geometry():
    from elastic.qha_core import strain_matrix

    cell = np.array([[3.1, 0.0, 0.0], [0.8, 2.7, 0.0], [-0.3, 0.7, 4.0]])
    engineering_xy = 0.012
    gradient = strain_matrix([0.0, 0.0, 0.0, 0.0, 0.0, engineering_xy])
    deformed_cell = cell @ gradient.T
    recovered_gradient = np.linalg.solve(cell, deformed_cell).T
    measured = (recovered_gradient.T @ recovered_gradient - np.eye(3)) / 2
    np.testing.assert_allclose(2 * measured[0, 1], engineering_xy, atol=1e-12)
    # For an exact composition, the requested sublattice occupancies must be integers.
    from fractions import Fraction
    for sites in (64, 80):
        ti = sites * Fraction(7, 10)
        print("sites =", sites, "requested Ti =", ti, "exactly representable =", ti.denominator == 1)
    print("recovered engineering xy strain =", round(2 * measured[0, 1], 6))


def parity():
    from matbench.compare_records import compare

    cpu = {"atoms": 2, "weights": {"model": "demo"}, "energy_eV": -5.0,
           "force_eV_A": [[0.1, 0.0, 0.0], [-0.1, 0.0, 0.0]],
           "stress_GPa": [[0.0] * 3 for _ in range(3)]}
    gpu = {**cpu, "energy_eV": -4.9999}
    errors = compare(cpu, gpu)
    assert math.isclose(errors["energy_eV_atom"], 5e-5, abs_tol=1e-12)
    print("maximum absolute discrepancies =", errors)


LABS = {"motifs": motifs, "viscosity": viscosity, "formulas": formulas,
        "elasticity": elasticity, "geometry": geometry, "parity": parity}

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("topic", choices=[*LABS, "all"])
    selected = parser.parse_args().topic
    for name, lab in LABS.items():
        if selected in (name, "all"):
            print(f"[{name}]")
            lab()
