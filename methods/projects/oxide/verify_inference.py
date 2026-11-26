"""Replay the archived test inputs through the shipped user-facing CSV predictor."""
import argparse
import csv
import json
from pathlib import Path
from tempfile import TemporaryDirectory

import h5py
import numpy as np

from oxide_basis import MOLAR_MASS
from predict_viscosity import predict_table


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('hdf', type=Path)
    args = parser.parse_args()
    with h5py.File(args.hdf) as f:
        names = f['X_columns'].asstr()[:12].tolist()
        x = f['X_test'][:, :12]
        t, p = (f[k][:].ravel() for k in ('T_test', 'P_test'))
    comparison = Path('results/test_comparison.csv')
    with comparison.open() as stream:
        ref = list(csv.DictReader(stream))
    with TemporaryDirectory() as tmp:
        table = Path(tmp)/'test.csv'

        def replay(oxide, temperature, basis):
            with table.open('w', newline='') as stream:
                writer = csv.writer(stream)
                writer.writerow(['T_K','P_GPa',*names])
                writer.writerows(np.column_stack([temperature,p[:len(oxide)],oxide]))
            return np.array([row[1:] for row in predict_table(table, basis)])

        mol = replay(x, t, 'mol')
        warmed = replay(x, t+1, 'mol')
        wt = replay(x[:20] * np.array([MOLAR_MASS[k] for k in names]), t[:20], 'wt')
    names = ['gp1','gp_mean','curve','blend']
    expected = np.array([[float(r[k]) for k in names] for r in ref])
    error = np.max(np.abs(mol[:,:4] - expected), axis=0)
    np.testing.assert_allclose(mol[:,:4], expected, atol=5e-5, rtol=0)
    np.testing.assert_allclose(mol[:20,:4], wt[:,:4], atol=5e-5, rtol=0)
    increase = warmed[:, :4] - mol[:, :4]
    outcome = {'test_rows':len(mol), 'max_abs_log10_difference_from_saved_test_predictions':dict(zip(names, error)),
               'input_units_max_difference_log10_first_20':float(np.max(abs(wt[:,:4] - mol[:20,:4]))),
               'warming_1K_increases':dict(zip(names, (increase > 0).sum(axis=0).tolist())),
               'warming_1K_max_increase_log10':dict(zip(names, increase.max(axis=0)))}
    print(json.dumps(outcome, indent=2))
    Path('results/inference_checks.json').write_text(json.dumps(outcome, indent=2)+'\n')


if __name__ == '__main__':
    main()
