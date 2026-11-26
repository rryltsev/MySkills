"""Replay the three GPVisc v0.3.4 checkpoints and export a small CPU predictor.

The conditioning calculation streams GPyTorch's inducing-point covariance
R R' + D in blocks. D includes the SGPR diagonal correction and observation
noise. Linear algebra is done in float64 to limit cancellation on CPU.
"""

import argparse
import csv
from pathlib import Path

import numpy as np
from scipy.linalg import cho_solve, cholesky, solve_triangular
from scipy.spatial.distance import cdist
from scipy.special import ndtr
from threadpoolctl import threadpool_limits


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_RELEASE = ROOT / "upstream" / "charlesll-gpvisc-5dbe6cc"
DEFAULT_POSTERIOR = ROOT / "results" / "gpvisc_posterior.npz"
SCALE = 4.4408
BLOCK = 256
NN_KEYS = ("A", "linears.0.weight", "linears.0.bias", "linears.1.weight",
           "linears.1.bias", "out.weight", "out.bias")


def inputs(composition, pressure_gpa, temperature_k):
    comp = np.asarray(composition, dtype=float)
    if comp.ndim == 1:
        comp = comp[None, :]
    if comp.ndim != 2 or comp.shape[1] != 12:
        raise ValueError("Expected 12 molar fractions in GPVisc oxide order.")
    pressure = np.broadcast_to(np.asarray(pressure_gpa).reshape(-1), (len(comp),))
    temperature = np.broadcast_to(np.asarray(temperature_k).reshape(-1), (len(comp),))
    if np.any(temperature <= 0) or np.any(pressure < 0) or np.any(comp < 0):
        raise ValueError("Require positive T in K, nonnegative P in GPa and mole fractions.")
    if not np.allclose(comp.sum(axis=1), 1., atol=1e-4, rtol=0):
        raise ValueError("Expected compositions normalized to one.")
    x = np.column_stack((1000 / temperature, pressure / 30, comp)).astype("float32")
    if not np.isfinite(x).all():
        raise ValueError("All model inputs must be finite.")
    return x


def chunks(x):
    for start in range(0, len(x), BLOCK):
        yield slice(start, min(start + BLOCK, len(x)))


def mean_network(x, weights):
    """Archived GELU network followed by the VFT equation, in log10 Pa s."""
    h = x[:, 1:]
    for i in (0, 1):
        h = h @ weights[f"linears.{i}.weight"].T + weights[f"linears.{i}.bias"]
        h = h * ndtr(h)
    b, c = np.exp(h @ weights["out.weight"].T + weights["out.bias"]).T
    return weights["A"] + b / (1000 / x[:, 0] - c)


def cross_kernel(x, weights):
    dist = cdist(x / weights["lengthscale"], weights["scaled_centres"])
    d = np.sqrt(5.) * dist
    return weights["outputscale"] * (1 + d + d * d / 3.) * np.exp(-d)


def load_tensors(file):
    import torch
    return torch.load(file, weights_only=True, map_location="cpu")


def condition(directory, expected_x, expected_y, with_variance):
    # Verify the archived conditioning set against the public HDF split.
    xt = load_tensors(directory / "X_train_valid.pth").numpy()
    yt = load_tensors(directory / "Y_train_valid.pth").numpy()
    np.testing.assert_allclose(xt, expected_x, rtol=1e-6, atol=1e-7)
    np.testing.assert_allclose(yt.ravel(), expected_y / SCALE, rtol=1e-6, atol=1e-7)
    state = {k: v.numpy() for k, v in load_tensors(directory / "gp.pth").items()}
    w = {k: state["mean_f." + k] for k in NN_KEYS}
    # Positive and GreaterThan constraints in the published state dict.
    def positive(key):
        return np.logaddexp(0., state[key].astype(float)) + state[key + "_constraint.lower_bound"]
    w["outputscale"] = positive("base_covar_module.raw_outputscale")
    w["lengthscale"] = positive("base_covar_module.base_kernel.raw_lengthscale").ravel()
    w["noise"] = positive("likelihood.noise_covar.raw_noise")
    centres = state["covar_module.inducing_points"]
    w["scaled_centres"] = centres / w["lengthscale"]
    kernel = cross_kernel(centres, w)
    factor = cholesky(kernel, lower=True)
    n = len(centres)
    info = np.eye(n)
    rhs = np.zeros(n)
    for s in chunks(xt):
        r = solve_triangular(factor, cross_kernel(xt[s], w).T, lower=True)
        diag = np.maximum(0., w["outputscale"] - np.einsum("ij,ij->j", r, r)) + w["noise"]
        scaled = r / np.sqrt(diag)
        info += scaled @ scaled.T
        rhs += r @ ((yt[s].ravel() - mean_network(xt[s], w) / SCALE) / diag)
    info_factor = cholesky(info, lower=True)
    beta = cho_solve((info_factor, True), rhs)
    w["coefficient"] = solve_triangular(factor.T, beta, lower=False)
    if with_variance:
        w["kernel_cholesky"] = factor
        w["information_cholesky"] = info_factor
    # Notebook 7 passes the full conditioning tensor back to predict at once.
    # In GPyTorch this triggers the 'equal x1,x2' diagonal-correction branch.
    train_pred = np.empty(len(xt))
    for s in chunks(xt):
        k = cross_kernel(xt[s], w)
        r = solve_triangular(factor, k.T, lower=True)
        correction = np.maximum(0., w["outputscale"] - np.einsum("ij,ij->j", r, r))
        pred = mean_network(xt[s], w) + SCALE * (k @ w["coefficient"])
        train_pred[s] = pred + correction / (correction + w["noise"]) * (yt[s].ravel() * SCALE - pred)
    return w, train_pred


def predict_one(x, weights, return_std):
    pred, grey, std = (np.empty(len(x)) for _ in range(3))
    for s in chunks(x):
        k = cross_kernel(x[s], weights)
        grey[s] = mean_network(x[s], weights)
        pred[s] = grey[s] + SCALE * (k @ weights["coefficient"])
        if return_std:
            r = solve_triangular(weights["kernel_cholesky"], k.T, lower=True)
            posterior_r = solve_triangular(weights["information_cholesky"], r, lower=True)
            var = weights["outputscale"] + weights["noise"] - np.sum(r*r, axis=0) + np.sum(posterior_r*posterior_r, axis=0)
            std[s] = SCALE * np.sqrt(np.maximum(0., var))
    return pred, grey, std


def predict_checkpoints(composition, pressure_gpa, temperature_k, posterior=DEFAULT_POSTERIOR):
    """Return GP1/2/3, their equal-weight mean, ANN1 and GP1 predictive SD."""
    x = inputs(composition, pressure_gpa, temperature_k)
    result = {}
    with np.load(posterior, allow_pickle=False) as data, threadpool_limits(limits=2):
        for i in (1, 2, 3):
            prefix = f"{i}/"
            weights = {k[len(prefix):]: data[k] for k in data.files if k.startswith(prefix)}
            pred, grey, std = predict_one(x, weights, i == 1)
            result[f"gp{i}"] = pred
            if i == 1:
                result["greybox1"], result["gp1_std"] = grey, std
    result["gp_mean"] = np.mean([result[f"gp{i}"] for i in (1, 2, 3)], axis=0)
    return result


def write_dicts(path, rows):
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=rows[0])
        writer.writeheader()
        writer.writerows(rows)


def run(release, out):
    import h5py
    out.mkdir(exist_ok=True, parents=True)
    model_dir = release / "src/gpvisc/models"
    hdf = release / "code_paper_EPSL/additional_data/all_viscosity.hdf5"
    with h5py.File(hdf, "r") as f:
        splits = ("train", "valid", "test")
        def concatenate(key):
            return np.concatenate([f[f"{key}_{s}"][:] for s in splits])
        x = inputs(concatenate("X")[:, :12], concatenate("P"), concatenate("T"))
        y = concatenate("y").ravel()
        sci = concatenate("Sciglass").ravel().astype(bool)
        split = np.concatenate([np.repeat(s, len(f[f"y_{s}"])) for s in splits])
        row = np.concatenate([np.arange(len(f[f"y_{s}"])) for s in splits])
    fit = split != "test"
    predictions, archive = {}, {}
    metrics = []
    for i in (1, 2, 3):
        print("Conditioning published GP", i, flush=True)
        w, train_pred = condition(model_dir / f"GP_model{i}", x[fit], y[fit], i == 1)
        archive.update({f"{i}/{k}": v for k, v in w.items()})
        pred, grey, std = predict_one(x, w, i == 1)
        pred[fit] = train_pred
        predictions[f"gp{i}"] = pred
        if i == 1:
            predictions["greybox1"], predictions["gp1_std"] = grey, std
    predictions["gp_mean"] = np.mean([predictions[f"gp{i}"] for i in (1, 2, 3)], axis=0)
    np.savez_compressed(out / "gpvisc_posterior.npz", **archive)
    write_dicts(out / "gpvisc_predictions.csv", [
        dict(split=split[j], row=row[j], y=y[j], sciglass=int(sci[j]),
             **{key: value[j] for key, value in predictions.items()}) for j in range(len(y))])
    for name, values in predictions.items():
        if name == "gp1_std":
            continue
        for dataset, mask in (("train_valid", fit), ("test", ~fit), ("test_star", ~fit & ~sci)):
            err = values[mask] - y[mask]
            record = dict(model=name, dataset=dataset, n=len(err),
                          rmse=np.sqrt(np.mean(err**2)), median_ae=np.median(abs(err)),
                          mean_ae=np.mean(abs(err)), r2=1-np.mean(err**2)/np.var(y[mask]))
            record["coverage_95"] = np.mean(abs(err) <= 1.959963984540054 * predictions["gp1_std"][mask]) if name == "gp1" and dataset != "train_valid" else ""
            metrics.append(record)
    write_dicts(out / "gpvisc_metrics.csv", metrics)
    print("GP RMSE:", [(m["model"], m["dataset"], round(m["rmse"], 5)) for m in metrics], flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--release", type=Path, default=DEFAULT_RELEASE)
    parser.add_argument("--out", type=Path, default=DEFAULT_POSTERIOR.parent)
    args = parser.parse_args()
    with threadpool_limits(limits=2):
        run(args.release, args.out)
