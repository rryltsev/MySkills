"""Sensitivity of the congruent-temperature estimate to finite-cell extrapolation."""
import argparse
import json
from pathlib import Path

import numpy as np

def main():
    cli = argparse.ArgumentParser(description=__doc__)
    cli.add_argument("samples", type=Path, nargs=3, help="one vertex summary JSON for each of three sizes")
    cli.add_argument("--out", type=Path, required=True)
    args = cli.parse_args()
    data = sorted((json.loads(path.read_text(encoding="utf8")) for path in args.samples), key=lambda row: row["N"])
    n=np.array([z["N"] for z in data])
    if np.any(n <= 0) or len(np.unique(n)) != 3:
        cli.error("three different positive cell sizes are required")
    t=np.array([z["central"]["T_K"] for z in data])
    rng=np.random.default_rng(4023129)
    draws=[]
    for z in data:
        store=z["replicates"]
        ts=np.array([v["T_K"] if isinstance(v,dict) else v[z["bootstrap_columns"].index("T_K")] for v in store])
        if len(ts)<16 or not z["central"]["profile"]:
            raise ValueError("Insufficient sampled roots or missing intervening order valley")
        draws.append(rng.choice(ts,10000,replace=True))
    boot=np.stack(draws)
    fits=[]
    for exponent,which in [(1.,np.arange(3)),(1.,np.arange(1,3)),(2/3,np.arange(3))]:
        v=n[which].astype(float)**(-exponent)
        design=np.column_stack([np.ones(len(v)),v/v.mean()])
        weights=np.linalg.pinv(design)[0]
        b=weights@boot[which]
        fits.append(dict(inverse_size_exponent=exponent,cells=n[which].tolist(),
                         intercept_K=float(weights@t[which]),
                         conditional_bootstrap_025_50_975_K=np.quantile(b,[.025,.5,.975]).tolist()))
    cells=[]
    for z in data:
        c=z["central"]; B=c["profile"]["lower_barrier"]
        cells.append(dict(N=z["N"],vertex_K=c["T_K"],x_at_pct=c["x_atpct"],jump_Q=c["jump_Q"],
                          heat_meV_atom=c["latent_meV_atom"],barrier_kBT=B,barrier_per_N23=B/z["N"]**(2/3)))
    result=dict(cells=cells,estimates=fits,source_files=[p.name for p in args.samples],
                note="Unweighted fits on three sizes or the two largest. The bulk intercept depends on the model "
                     "for finite-size corrections. The resampling intervals omit inter-source equilibration "
                     "scatter, the order divider, cell shape and the displacement wall. The inverse-N^(2/3) "
                     "fit tests sensitivity only. Barrier values refer to the smaller peak at the 4:1 vertex.")
    with args.out.open("x", encoding="utf-8") as out:
        out.write(json.dumps(result,indent=2,allow_nan=False)+"\n")


if __name__=="__main__":
    main()
