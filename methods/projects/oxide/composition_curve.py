"""Train or use a smooth MYEGA-type model on the archived GPVisc splits.

For a prediction use a CSV with T (K), P (GPa) and all twelve oxide
columns in mole fractions, or all in mol%. The output is log10(eta/Pa s).
"""
import argparse
import csv
import json
import os
from pathlib import Path

os.environ["OPENBLAS_NUM_THREADS"] = os.environ.get("VISCOSITY_THREADS", "3")
os.environ["MKL_NUM_THREADS"] = os.environ["OPENBLAS_NUM_THREADS"]

import numpy as np
from scipy.linalg import cho_factor, cho_solve, solve_triangular
from scipy.spatial import cKDTree
from scipy.spatial.distance import cdist


OXIDES = "sio2 tio2 al2o3 feo fe2o3 mno mgo cao na2o k2o p2o5 h2o".split()
ASYMPTOTE = -4.7
RESULTS = Path(__file__).resolve().parents[1] / "results"


def read_split(path, split):
    import h5py

    with h5py.File(path, "r") as data:
        published = data["X_columns"].asstr()[:12].tolist()
        order = [published.index(name) for name in OXIDES]
        x = data[f"X_{split}"][:, :12][:, order].astype(float)
        p = data[f"P_{split}"][:].ravel()
        t = data[f"T_{split}"][:].ravel()
        y = data[f"y_{split}"][:].ravel()
        sciglass = data[f"Sciglass_{split}"][:].ravel().astype(bool)
    return x, p, t, y, sciglass


def coordinates(x, pressure):
    # Stoichiometric charge-balance descriptors help resolve the transition
    # between modifier-rich and alumina-rich melts. The array uses OXIDES order.
    x = np.asarray(x, dtype=float)
    if x.ndim != 2 or x.shape[1] != len(OXIDES) or not np.isfinite(x).all():
        raise ValueError("Expected twelve finite oxide amounts per sample")
    if (x < 0).any() or (x.sum(axis=1) <= 0).any():
        raise ValueError("Molar compositions require nonnegative amounts and a positive sum")
    x = x / x.sum(axis=1, keepdims=True)
    si, ti, al, fe, f3, mn, mg, ca, na, k, ph, water = x.T
    alkali = na + k
    surplus = fe + mn + mg + ca + alkali - al - f3
    pressure = np.broadcast_to(np.asarray(pressure, dtype=float), (len(x),))
    if (pressure < 0).any() or not np.isfinite(pressure).all():
        raise ValueError("Pressure must be finite and nonnegative in GPa")
    return np.column_stack([
        x, 0.6 * np.sqrt(water + 0.0005), 0.5 * np.maximum(surplus, 0),
        0.5 * np.maximum(alkali - al, 0), 0.12 * np.arcsinh(pressure / 1.5)
    ])


def landmarks(states, count):
    # Deterministic farthest-point sampling covers rare chemistries and pressures.
    nearest = np.sum((states - np.mean(states, axis=0)) ** 2, axis=1)
    index = int(np.argmin(nearest))
    chosen = []
    nearest[:] = np.inf
    for _ in range(min(count, len(states))):
        chosen.append(index)
        dist = np.sum((states - states[index]) ** 2, axis=1)
        np.minimum(nearest, dist, out=nearest)
        index = int(np.argmax(nearest))
    return states[chosen]


def basis(states, centers, length):
    distance = np.sqrt(5.0) / length * cdist(states, centers)
    kernel = (1 + distance + distance**2 / 3) * np.exp(-distance)
    return np.column_stack([np.ones(len(states)), states, kernel])


def temperature_terms(temperature, order):
    temperature = np.asarray(temperature, dtype=float)
    if np.any(temperature <= 0) or not np.isfinite(temperature).all():
        raise ValueError("Temperature must be positive and finite in K")
    inverse = 1000.0 / temperature
    z = inverse - 0.8
    terms = [np.ones_like(z), z]
    if order == 3:
        terms.append(0.4 * z**2)
    return inverse, np.column_stack(terms)


def setup_basis(states, centers, length):
    phi = basis(states, centers, length)
    start = 1 + states.shape[1]
    chol = np.linalg.cholesky(basis(centers, centers, length)[:, start:] + 1e-8 * np.eye(len(centers)))
    phi[:, start:] = solve_triangular(chol, phi[:, start:].T, lower=True).T
    return phi, chol, start


def equations(phi, groups, terms, weight, target, ridge):
    # Aggregate the normal equations for repeated temperatures at the same
    # pressure and composition, without constructing a full row-wise design.
    width = phi.shape[1]
    size = len(phi)
    blocks = terms.shape[1]
    lhs = np.zeros((width * blocks, width * blocks))
    rhs = np.zeros(width * blocks)
    for i in range(blocks):
        section = slice(i * width, (i + 1) * width)
        w = weight * terms[:, i]
        rhs[section] = phi.T @ np.bincount(groups, weights=w * target, minlength=size)
        for j in range(i + 1):
            other = slice(j * width, (j + 1) * width)
            moment = np.bincount(groups, weights=w * terms[:, j], minlength=size)
            block = phi.T @ (phi * moment[:, None])
            lhs[section, other] = block
            if j != i:
                lhs[other, section] = block.T
    lhs.flat[::len(lhs) + 1] += ridge
    return cho_solve(cho_factor(lhs, lower=True, check_finite=False), rhs, check_finite=False).reshape(blocks, width).T


def curve(raw, inverse):
    # Protect the exponential against overflow during optimization.
    return ASYMPTOTE + inverse * np.exp(np.clip(raw, -30, 30))


def fit(x, p, t, y, length, count, order, ridge):
    states, group = np.unique(coordinates(x, p), axis=0, return_inverse=True)
    centers = landmarks(states, count)
    phi, chol, start = setup_basis(states, centers, length)
    inverse, terms = temperature_terms(t, order)
    amplitude = y - ASYMPTOTE
    if np.min(amplitude) <= 0:
        raise ValueError("Training target at or below the high-temperature asymptote")
    target = np.log(amplitude / inverse)
    weight = amplitude**2
    coef = equations(phi, group, terms, weight, target, ridge)
    # One robust Gauss–Newton refinement in the measured log10 viscosity.
    raw = np.sum((phi @ coef)[group] * terms, axis=1)
    predicted = curve(raw, inverse)
    amplitude = np.maximum(predicted - ASYMPTOTE, 1e-6)
    error = predicted - y
    w = amplitude**2 / np.maximum(1, np.abs(error) / 0.6)
    coef = equations(phi, group, terms, w, raw - error / amplitude, ridge)
    coef[start:] = solve_triangular(chol.T, coef[start:], lower=False)
    return dict(centers=centers, length=length, coefficients=coef)


def predict_curve(model, x, p, t):
    states, group = np.unique(coordinates(x, p), axis=0, return_inverse=True)
    coef = model["coefficients"]
    inverse, terms = temperature_terms(t, coef.shape[1])
    parameters = basis(states, model["centers"], float(model["length"])) @ coef
    return curve(np.sum(parameters[group] * terms, axis=1), inverse)


def residual_coordinates(x, p, t, temperature_scale):
    return np.column_stack([coordinates(x, p), temperature_scale * 1000.0 / t])


def add_correction(model, x, p, t, y, temperature_scale, length, noise):
    prior = predict_curve(model, x, p, t)
    scale = np.maximum(1.0, (prior - ASYMPTOTE) / 5)
    model.update(support=residual_coordinates(x, p, t, temperature_scale),
                 residual=(y - prior) / scale, temperature_scale=temperature_scale,
                 residual_length=length, noise=noise)


def local_residual(model, x, p, t):
    support = model["support"]
    scaled = residual_coordinates(x, p, t, float(model["temperature_scale"]))
    distance, neighbor = cKDTree(support).query(scaled, k=min(80, len(support)))
    coefficient = np.sqrt(5) / float(model["residual_length"])
    noise = float(model["noise"])
    residual = model["residual"]
    out = np.zeros(len(scaled))
    for i, ids in enumerate(neighbor):
        subset = support[ids]
        a = coefficient * cdist(subset, subset)
        k = (1 + a + a**2 / 3) * np.exp(-a)
        k.flat[::len(subset) + 1] += noise
        b = coefficient * distance[i]
        cross = (1 + b + b**2 / 3) * np.exp(-b)
        weights = cho_solve(cho_factor(k, lower=True, check_finite=False),
                            cross, check_finite=False)
        out[i] = weights @ residual[ids]
    return out


def predict(model, x, p, t):
    prior = predict_curve(model, x, p, t)
    if "support" in model:
        prior += np.maximum(1.0, (prior - ASYMPTOTE) / 5) * local_residual(model, x, p, t)
    return prior


def measures(y, prediction):
    error = y - prediction
    return {"n": int(len(y)), "rmse": float(np.sqrt(np.mean(error**2))),
            "median_ae": float(np.median(np.abs(error))), "mean_ae": float(np.mean(np.abs(error))),
            "r2": float(1 - error @ error / np.sum((y - y.mean())**2))}


def write_json(path, data):
    path.write_text(json.dumps(data, indent=2) + "\n", encoding="utf-8")


def select_arrays(training, validation):
    xv, pv, tv, yv = validation[:4]
    scores = []
    for length in (0.22, 0.46):
        for order in (2, 3):
            for ridge in (0.08, 0.8):
                settings = dict(count=480, length=length, order=order, ridge=ridge)
                model = fit(*training[:4], **settings)
                prediction = predict(model, xv, pv, tv)
                scores.append(dict(settings=settings, validation=measures(yv, prediction)))
                print(scores[-1], flush=True)
    winner = min(scores, key=lambda row: row["validation"]["rmse"])
    summary = {"candidates": scores, "selected": winner}
    x, p, t, y = training[:4]
    model = fit(x, p, t, y, **summary["selected"]["settings"])
    scores = [{"settings": None, "validation": summary["selected"]["validation"]}]
    for temperature_scale in (0.45, 0.9):
        for length in (0.12, 0.3):
            for noise in (0.012, 0.08):
                settings = dict(temperature_scale=temperature_scale, length=length, noise=noise)
                add_correction(model, x, p, t, y, **settings)
                prediction = predict(model, xv, pv, tv)
                scores.append(dict(settings=settings, validation=measures(yv, prediction)))
                print(scores[-1], flush=True)
    summary["corrections"] = scores
    summary["selected_correction"] = min(scores, key=lambda row: row["validation"]["rmse"])
    return summary


def select(path):
    summary = select_arrays(read_split(path, "train"), read_split(path, "valid"))
    write_json(RESULTS / "composition_curve_validation.json", summary)


def evaluate(path):
    chosen = json.loads((RESULTS / "composition_curve_validation.json").read_text())
    train = read_split(path, "train")
    valid = read_split(path, "valid")
    merged = [np.concatenate([a, b]) for a, b in zip(train, valid)]
    model = fit(*merged[:4], **chosen["selected"]["settings"])
    correction = chosen["selected_correction"]["settings"]
    if correction is not None:
        add_correction(model, *merged[:4], **correction)
    model_path = RESULTS / "composition_curve_model.npz"
    np.savez_compressed(model_path, **model)
    with np.load(model_path, allow_pickle=False) as archive:
        saved = dict(archive)
        train_prediction = predict(saved, *merged[:3])
        x, p, t, y, sciglass = read_split(path, "test")
        pred = predict(saved, x, p, t)
        warming = predict(saved, x, p, t + 1) - pred
    metrics = {"train_valid": measures(merged[3], train_prediction), "test": measures(y, pred),
               "test_curated": measures(y[~sciglass], pred[~sciglass]),
               "warming_by_1K_on_test": {"increases": int(np.sum(warming > 0)),
                                         "curated_increases": int(np.sum(warming[~sciglass] > 0)),
                                         "max_increase_log10": float(np.max(warming))}}
    write_json(RESULTS / "composition_curve_metrics.json", metrics)
    with (RESULTS / "composition_curve_predictions.csv").open("w", encoding="utf-8", newline="") as stream:
        writer = csv.writer(stream)
        writer.writerow(["test_row", "log10_eta", "pred_log10_eta", "sciglass"])
        writer.writerows(zip(range(len(y)), y, pred, sciglass.astype(int)))
    print(json.dumps(metrics, indent=2))


def predict_csv(path, model_path):
    with path.open(encoding="utf-8-sig", newline="") as stream:
        rows = list(csv.DictReader(stream))
    x = np.asarray([[float(row[k]) for k in OXIDES] for row in rows])
    p = np.asarray([float(row["P"]) for row in rows])
    t = np.asarray([float(row["T"]) for row in rows])
    with np.load(model_path, allow_pickle=False) as model:
        predicted = predict(model, x, p, t)
    print("pred_log10_eta")
    for value in predicted:
        print(f"{value:.10g}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("phase", choices=("select", "evaluate", "predict"))
    parser.add_argument("path", type=Path, help="GPVisc HDF5 for training, input CSV for prediction")
    parser.add_argument("--model", type=Path, default=RESULTS / "composition_curve_model.npz")
    args = parser.parse_args()
    if args.phase == "predict":
        predict_csv(args.path, args.model)
    else:
        RESULTS.mkdir(exist_ok=True)
        (select if args.phase == "select" else evaluate)(args.path)
