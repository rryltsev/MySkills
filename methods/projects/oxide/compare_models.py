"""Compare paired predictions on the archived test rows, with a composition bootstrap."""
import argparse
import csv
import json
from pathlib import Path

import h5py
import numpy as np

from oxide_basis import MOLAR_MASS


def subset_masks(x, columns, t, p, sciglass):
    mass = x * np.array([MOLAR_MASS[c] for c in columns])
    mass /= mass.sum(axis=1, keepdims=True)
    cmas = mass[:, [columns.index(c) for c in ('cao', 'mgo', 'al2o3', 'sio2')]].sum(axis=1)
    ca, si = mass[:, columns.index('cao')], mass[:, columns.index('sio2')]
    proxy = ((p <= 0.001) & (t >= 1473) & (t <= 1973) & (cmas >= 0.85)
             & (si > 0) & (ca >= 0.8 * si) & (ca <= 1.5 * si))
    return {'all': np.ones(len(t), dtype=bool), 'curated': ~sciglass, 'sciglass': sciglass,
            'CMAS_proxy': proxy, 'pressure_above_0.001_GPa': p > 0.001}


def read_rows(path):
    with path.open(encoding='utf-8-sig', newline='') as f:
        return list(csv.DictReader(f))


def values(rows, col):
    return np.array([float(r[col]) for r in rows])


def compare(y, group, predictions, mask, seed=7729, replicates=5000):
    y = y[mask]
    groups, inverse = np.unique(group[mask], return_inverse=True)
    counts = np.bincount(inverse)
    errors = {name: pred[mask] - y for name, pred in predictions.items()}
    rng = np.random.default_rng(seed)
    draw = rng.integers(len(groups), size=(replicates, len(groups)))
    total = counts[draw].sum(axis=1)
    samples = {}
    result = {}
    for name, error in errors.items():
        sse = np.bincount(inverse, weights=error**2)
        samples[name] = np.sqrt(sse[draw].sum(axis=1) / total)
        result[name] = dict(n=len(y), groups=len(groups), rmse=float(np.sqrt(np.mean(error**2))),
                            medae=float(np.median(np.abs(error))), mean_ae=float(np.mean(np.abs(error))),
                            r2=float(1 - sum(error**2) / sum((y-y.mean())**2)) if np.ptp(y) else None)
    for name, bootstrap in samples.items():
        delta = bootstrap - samples['gp1']
        result[name]['delta_rmse_vs_gp1'] = result[name]['rmse'] - result['gp1']['rmse']
        result[name]['delta_95pct_percentile_interval'] = np.quantile(delta, [0.025, 0.975]).tolist()
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('hdf', type=Path)
    parser.add_argument('--results', type=Path, default=Path('results'))
    args = parser.parse_args()
    with h5py.File(args.hdf) as f:
        y, t, p, sciglass, group = [f[k+'_test'][:].ravel() for k in ('y', 'T', 'P', 'Sciglass', 'Name')]
        columns = [s.decode().lower() for s in f['X_columns'][:12].ravel()]
        x = f['X_test'][:, :12]
    masks = subset_masks(x, columns, t, p, sciglass.astype(bool))
    gp = [r for r in read_rows(args.results/'gpvisc_predictions.csv') if r['split']=='test']
    curve = read_rows(args.results/'composition_curve_predictions.csv')
    for rows, idx, obs in ((gp, 'row', 'y'), (curve, 'test_row', 'log10_eta')):
        np.testing.assert_array_equal(values(rows, idx), np.arange(len(y)))
        np.testing.assert_allclose(values(rows, obs), y, rtol=1e-6, atol=1e-6)
    predictions = {k: values(gp, k) for k in ('gp1', 'gp2', 'gp3')}
    predictions['gp_mean'] = np.mean([predictions[k] for k in ('gp1','gp2','gp3')], axis=0)
    predictions['curve'] = values(curve, 'pred_log10_eta')
    predictions['blend'] = (predictions['gp_mean'] + predictions['curve']) / 2
    coverage = {}
    with h5py.File(args.hdf) as f:
        for split in ('train','valid','test'):
            ts, ps, sg, names = [f[k+'_'+split][:].ravel() for k in ('T','P','Sciglass','Name')]
            proxy = subset_masks(f['X_'+split][:, :12], columns, ts, ps, sg.astype(bool))['CMAS_proxy']
            coverage[split] = {'rows': int(proxy.sum()), 'groups':int(len(np.unique(names[proxy])))}
    provenance = args.results / 'test_curated_sources.csv'
    if provenance.exists():
        origin = read_rows(provenance)
        np.testing.assert_array_equal(values(origin, 'row'), np.flatnonzero(~sciglass.astype(bool)))
        for key, field in [('Iida_2000', 'iida_definite'), ('slag_in_source_title', 'slag_title_definite')]:
            selection = np.zeros(len(y), dtype=bool)
            for row in origin:
                selection[int(row['row'])] = row[field].lower() in ('true','1')
            masks[key] = selection
    report = {'bootstrap': {'groups': 'archived Name_test', 'replicates': 5000, 'seed': 7729},
              'cmas_proxy_coverage': coverage,
              'metrics': {k: compare(y, group, predictions, mask) for k, mask in masks.items() if mask.any()}}
    (args.results/'comparison.json').write_text(json.dumps(report, indent=2)+'\n', encoding='utf-8')
    with (args.results/'test_comparison.csv').open('w', encoding='utf-8', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(['row','composition_group','T_K','P_GPa','log10_eta','sciglass','CMAS_proxy',*predictions])
        writer.writerows(zip(range(len(y)), (g.decode() for g in group), t, p, y, sciglass.astype(int),
                             masks['CMAS_proxy'].astype(int), *predictions.values()))
    for subset, metrics in report['metrics'].items():
        print(subset, metrics['gp1']['n'], {m: round(d['rmse'], 4) for m,d in metrics.items()})


if __name__ == '__main__':
    main()
