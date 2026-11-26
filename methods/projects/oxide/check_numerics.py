"""Check the numerical contracts of the oxide workflows without external data."""
from collections import defaultdict

import numpy as np

from audit_dataset import IIDA, audit_split, row_key
from classical_benchmarks import scores
from iida_transfer import composition_key, linear_control
from reproduce_gpvisc import SCALE, cross_kernel, inputs, predict_one


def posterior():
    composition = np.eye(12)[:1]
    x = inputs(composition, 3, 1600)
    np.testing.assert_allclose(x[0, :2], [0.625, 0.1])
    # One inducing point, constant neural mean, no trained or published weights.
    w = {
        "scaled_centres": x.copy(), "lengthscale": np.ones(14), "outputscale": 1.0,
        "noise": 0.25, "A": -2.0, "linears.0.weight": np.zeros((1, 13)),
        "linears.0.bias": np.zeros(1), "linears.1.weight": np.zeros((1, 1)),
        "linears.1.bias": np.zeros(1), "out.weight": np.zeros((2, 1)),
        "out.bias": np.log([1000.0, 100.0]), "coefficient": np.array([0.25]),
        "kernel_cholesky": np.ones((1, 1)), "information_cholesky": np.array([[2.0]]),
    }
    np.testing.assert_allclose(cross_kernel(x, w), [[1.0]])
    prediction, grey, std = predict_one(x, w, True)
    np.testing.assert_allclose(grey, [-2 + 1000/1500])
    np.testing.assert_allclose(prediction, grey + 0.25*SCALE)
    np.testing.assert_allclose(std, [SCALE*np.sqrt(0.5)])
    print("GP posterior and uncertainty agree with analytic one-centre values.")


def provenance():
    x = np.eye(12)[:1]
    np.testing.assert_equal(composition_key(x[0]), composition_key(100*x[0]))
    key = row_key(x[0], 0, 1600, 1.5)
    lookup = defaultdict(list, {key: [("lp:1", IIDA), ("lp:2", "synthetic source")]})
    data = {"x": np.repeat(x, 2, axis=0), "T": np.full(2, 1600.0), "P": np.zeros(2),
            "y": np.full(2, 1.5), "Sciglass": np.array([False, False]), "Name": np.array([b"A"]*2)}
    rows, keys, summary = audit_split(data, "test", lookup, set())
    assert keys == [key, key] and len(rows) == 2
    assert summary["multiple_possible_sources"] == summary["iida_possible"] == 2
    assert summary["iida_definite"] == 0
    print("Two identical measurements retained, both with ambiguous source attribution.")


def controls():
    train = (np.tile(np.eye(12)[0], (4, 1)), np.zeros(4), np.arange(1400.0, 1800, 100),
             np.array([3.0, 2.5, 2.0, 1.5]))
    held = (train[0][:2], np.zeros(2), np.array([1450.0, 1550.0]), np.array([2.75, 2.25]))
    predictions = linear_control(train, held)
    np.testing.assert_allclose(predictions, held[-1], atol=1e-13)
    assert scores(held[-1], predictions)["rmse"] < 1e-13
    print("Standardized linear transfer returns 2.75 and 2.25 on held temperatures.")


if __name__ == "__main__":
    posterior()
    provenance()
    controls()
