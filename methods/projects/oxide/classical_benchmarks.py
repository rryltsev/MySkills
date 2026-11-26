"""Refit Table 1 classical controls on the authors' saved split (notebook 4)."""
import argparse
import json
from pathlib import Path

import h5py
import numpy as np
import sklearn
from sklearn.linear_model import LinearRegression
from sklearn.neural_network import MLPRegressor
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVR
from threadpoolctl import threadpool_limits


def scores(observed, predicted):
    error = predicted - observed
    return dict(n=len(error), rmse=float(np.sqrt(np.mean(error**2))),
                medae=float(np.median(np.abs(error))),
                r2=float(1 - np.sum(error**2) / np.sum((observed - observed.mean())**2)))


def features(data, split):
    temp = data[f'T_{split}'][:].reshape(-1, 1)
    pressure = data[f'P_{split}'][:].reshape(-1, 1)
    return np.column_stack((temp, pressure, data[f'X_{split}'][:, :12]))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('hdf', type=Path)
    parser.add_argument('model', choices=['lr', 'xgb', 'ann', 'svr'])
    parser.add_argument('--out', type=Path, default=Path('results/classical'))
    args = parser.parse_args()
    estimators = {
        'lr': LinearRegression(),
        'ann': MLPRegressor(solver='adam', alpha=1e-3, hidden_layer_sizes=(200, 200),
                            activation='relu', max_iter=20000, random_state=167),
        'svr': SVR(kernel='rbf', C=298, epsilon=0.000219, cache_size=1000),
    }
    if args.model == 'xgb':
        from xgboost import XGBRegressor
        estimators['xgb'] = XGBRegressor(n_estimators=893, max_depth=9, eta=0.03186217323374842,
                                         subsample=0.9079053385183469, colsample_bytree=0.9473360174104577,
                                         n_jobs=1)
    with h5py.File(args.hdf, 'r') as data:
        x_fit = np.concatenate([features(data, s) for s in ('train', 'valid')])
        y_fit = np.concatenate([data[f'y_{s}'][:].ravel() for s in ('train', 'valid')])
        x_test = features(data, 'test')
        y_test = data['y_test'][:].ravel()
        curated = ~data['Sciglass_test'][:].ravel().astype(bool)
    scaler = StandardScaler().fit(x_fit)
    x_fit = scaler.transform(x_fit)
    x_test = scaler.transform(x_test)
    model = estimators[args.model]
    with threadpool_limits(limits=1):
        model.fit(x_fit, y_fit)
        p_fit = model.predict(x_fit)
        p_test = model.predict(x_test)
    metrics = {'train_valid': scores(y_fit, p_fit), 'test': scores(y_test, p_test),
               'test_star': scores(y_test[curated], p_test[curated]),
               'sklearn': sklearn.__version__}
    args.out.mkdir(parents=True, exist_ok=True)
    (args.out / f'{args.model}.json').write_text(json.dumps(metrics, indent=2) + '\n')
    print(args.model, metrics, flush=True)


if __name__ == '__main__':
    main()
