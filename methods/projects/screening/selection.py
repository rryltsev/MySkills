"""The single final-selection rule for NN, PySR and Operon.

Only prequalified candidates and noisy validation labels enter this function.
"""
import math
import numpy as np

def one_se(candidates,predictions,y,sy):
    if not candidates:raise ValueError('No admissible candidate')
    loss=((np.asarray(predictions,dtype=float)-np.asarray(y)[None,:])/sy)**2
    if not np.isfinite(loss).all():raise ValueError('Nonfinite validation loss')
    mse=np.mean(loss,axis=1)
    best=min(range(len(candidates)),key=lambda i:(mse[i],candidates[i]['selection_complexity'],candidates[i]['candidate_id']))
    se=float(np.std(loss[best],ddof=1)/math.sqrt(loss.shape[1]))
    threshold=float(mse[best]+se)
    eligible=[i for i in range(len(candidates)) if mse[i]<=threshold]
    selected=min(eligible,key=lambda i:(candidates[i]['selection_complexity'],mse[i],candidates[i]['candidate_id']))
    return selected,dict(rule='holdout_one_se_v2',reference_candidate_id=candidates[best]['candidate_id'],
        selected_candidate_id=candidates[selected]['candidate_id'],minimum_mse=float(mse[best]),
        se_mse=se,threshold_mse=threshold,n_admissible=len(candidates),n_within_band=len(eligible),
        selected_mse=float(mse[selected]),selected_val_nrmse=float(np.sqrt(mse[selected])),
        interpretation='heuristic, not a significance test or a CV standard error')
