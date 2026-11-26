"""Predict melt viscosity from named oxide contents, temperature (K) and pressure (GPa)."""
import argparse
import csv
import sys
from pathlib import Path

import numpy as np

from composition_curve import OXIDES as CURVE_ORDER, RESULTS, predict as predict_curve
from oxide_basis import MOLAR_MASS
from reproduce_gpvisc import predict_checkpoints


def predict_table(path, basis):
    with path.open(encoding='utf-8-sig', newline='') as stream:
        reader = csv.DictReader(stream)
        fields = [k.strip().lower() for k in reader.fieldnames or []]
        if len(set(fields)) != len(fields):
            raise ValueError('Duplicated column name')
        missing = {'t_k', 'p_gpa'} - set(fields)
        unknown = set(fields) - {'id', 't_k', 'p_gpa', *MOLAR_MASS}
        if missing or unknown or not set(fields).intersection(MOLAR_MASS):
            raise ValueError(f'Expected T_K, P_GPa and oxide column(s). Missing {sorted(missing)}, unknown {sorted(unknown)}')
        reader.fieldnames = fields
        rows = list(reader)
    if not rows or any(None in r for r in rows):
        raise ValueError('Expected a nonempty CSV with one value per named column')
    oxide = np.array([[float(r.get(k, 0)) for k in MOLAR_MASS] for r in rows])
    if not np.isfinite(oxide).all() or (oxide < 0).any() or (oxide.sum(axis=1) <= 0).any():
        raise ValueError('Oxide amounts must be finite, nonnegative and have a positive sum')
    if basis == 'wt':
        oxide /= np.array(list(MOLAR_MASS.values()))
    oxide /= oxide.sum(axis=1, keepdims=True)
    t = np.array([float(r['t_k']) for r in rows])
    p = np.array([float(r['p_gpa']) for r in rows])
    gp = predict_checkpoints(oxide, p, t)
    with np.load(RESULTS/'composition_curve_model.npz', allow_pickle=False) as saved:
        curve = predict_curve(dict(saved), oxide[:, [list(MOLAR_MASS).index(k) for k in CURVE_ORDER]], p, t)
    blend = (gp['gp_mean'] + curve) / 2
    with np.errstate(over='raise', invalid='raise'):
        eta = np.power(10., blend)
    return zip([r.get('id', str(i+1)) for i,r in enumerate(rows)], gp['gp1'], gp['gp_mean'], curve, blend, eta)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('csv', type=Path)
    parser.add_argument('--basis', choices=('mol', 'wt'), required=True,
                        help='Common molar or mass basis for all oxide amounts. Rows are normalized to one.')
    args = parser.parse_args()
    try:
        rows = predict_table(args.csv, args.basis)
    except (ValueError, TypeError, FloatingPointError) as error:
        parser.exit(2, f'Input or prediction error: {error}\n')
    writer = csv.writer(sys.stdout, lineterminator='\n')
    writer.writerow(['id','gp1_log10_eta','gp_mean_log10_eta','curve_log10_eta','blend_log10_eta','blend_eta_Pa_s'])
    writer.writerows(rows)


if __name__ == '__main__':
    main()
