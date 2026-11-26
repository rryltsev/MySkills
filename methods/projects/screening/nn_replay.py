import numpy as np
from scipy.special import expit


def nn_predict(path,x):
    with np.load(path,allow_pickle=False) as m:
        if set(m.files)!={'W0','W1','b0','b1','x_mean','x_scale','y_mean','y_scale'}:raise ValueError('Invalid weight schema')
        if m['W0'].shape[0]!=2 or m['W1'].shape!=(m['W0'].shape[1],1):raise ValueError('Not one hidden layer')
        if not all(np.isfinite(m[k]).all() for k in m.files) or np.any(m['x_scale']<=0) or np.any(m['y_scale']<=0):raise ValueError('Invalid weights/scaling')
        return ((expit(((x-m['x_mean'])/m['x_scale'])@m['W0']+m['b0'])@m['W1']+m['b1'])*m['y_scale']+m['y_mean']).ravel()
