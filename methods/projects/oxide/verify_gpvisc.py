"""Check the exported GP1 posterior against the archived native prediction API."""

import argparse
import json
from pathlib import Path
import sys
from unittest.mock import patch

import h5py
import numpy as np
import torch
from threadpoolctl import threadpool_limits

from reproduce_gpvisc import DEFAULT_POSTERIOR, DEFAULT_RELEASE, inputs, predict_checkpoints


def verify(release, posterior, sample_size):
    sys.path.insert(0, str(release / "src"))
    import gpvisc
    directory = release / "src/gpvisc/models/GP_model1"
    # load_gp_model initializes inducing points via K-means, then immediately
    # overwrites them from gp.pth. Supply those final centres at initialization.
    state = torch.load(directory / "gp.pth", map_location="cpu", weights_only=True)
    centres = state["covar_module.inducing_points"]
    with patch.object(gpvisc.models, "KMeans") as kmeans:
        kmeans.return_value.cluster_centers_ = centres.numpy()
        model, likelihood = gpvisc.load_gp_model(model_number=1, device="cpu")
    torch.testing.assert_close(model.covar_module.inducing_points, centres, rtol=0, atol=0)
    with h5py.File(release / "code_paper_EPSL/additional_data/all_viscosity.hdf5") as data:
        n = len(data["y_test"])
        idx = np.arange(n) if sample_size is None else np.unique(np.linspace(0, n - 1, sample_size, dtype=int))
        comp = data["X_test"][idx, :12]
        pressure = data["P_test"][idx].ravel()
        temperature = data["T_test"][idx].ravel()
        y = data["y_test"][idx].ravel()
        sci = data["Sciglass_test"][idx].ravel().astype(bool)
    x = torch.from_numpy(inputs(comp, pressure, temperature))
    gp, sd = gpvisc.predict(x, model, likelihood, device="cpu")
    grey = gpvisc.predict(x, model, likelihood, model_to_use="ann", device="cpu")
    exported = predict_checkpoints(comp, pressure, temperature, posterior)
    result = {"gpvisc_version": "0.3.4", "torch": torch.__version__, "n_rows": len(idx),
              "rows": "all" if sample_size is None else idx.tolist()}
    for key, mask in (("test", np.ones(len(y), dtype=bool)), ("test_star", ~sci)):
        error = gp[mask] - y[mask]
        result[key] = dict(n=int(mask.sum()), rmse=float(np.sqrt(np.mean(error**2))),
                           median_ae=float(np.median(np.abs(error))),
                           mean_ae=float(np.mean(np.abs(error))),
                           coverage_95=float(np.mean(abs(error) <= 1.959963984540054 * sd[mask])))
    for key, value in (("gp1", gp), ("gp1_std", sd), ("greybox1", grey)):
        delta = exported[key] - value
        result[key] = {"max_abs_diff": float(np.max(np.abs(delta))),
                       "rms_diff": float(np.sqrt(np.mean(delta ** 2)))}
    return result


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--release", type=Path, default=DEFAULT_RELEASE)
    p.add_argument("--posterior", type=Path, default=DEFAULT_POSTERIOR)
    p.add_argument("--sample-size", type=int, help="Use evenly spaced test rows for a fast spot check")
    args = p.parse_args()
    torch.set_num_threads(2)
    with threadpool_limits(limits=2):
        report = verify(args.release, args.posterior, args.sample_size)
    target = args.posterior.parent / "gpvisc_native_check.json"
    target.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(report)
