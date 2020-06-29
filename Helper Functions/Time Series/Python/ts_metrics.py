#Author: Justin Swansburg, Mark Philip

#Make sure you are connected to DataRobot Client.


#These functions can be used to calculate common evaluation metrics

import numpy as np


#####################
# Evaluation Metrics
#####################

def mae(act, pred, weight=None):
    """
    MAE = Mean Absolute Error = mean( abs(act - pred) )
    """
    if len(pred.shape) > 1:
        if pred.shape[1] == 2:
            pred = pred[:, 1]
        else:
            pred = pred.ravel()

    pred = pred.astype(np.float64, copy=False)
    d = act - pred
    ad = np.abs(d)
    if weight is not None:
        if weight.sum() == 0:
            return 0
        ad = ad * weight / weight.mean()
    mae = ad.mean()

    if np.isnan(mae):
        return np.finfo(np.float64).max
    else:
        return mae


def mape(act, pred, nan='ignore'):

    # ignore NAN (drop rows), do nothing, replace Nan with 0
    if nan not in ['ignore', 'set_to_zero', 'error']:
        raise ValueError(f'{nan} must be either ignore, set_to_zero, or error')

    act, pred = np.array(act), np.array(pred)
    pred = pred.astype(np.float64, copy=False)
    n = np.abs(act - pred)
    d = act
    ape = n / d

    if nan == 'set_to_zero':
        ape[~np.isfinite(ape)] = 0
    elif nan == 'ignore':
        ape = ape[np.isfinite(ape)]

    smape = np.mean(ape)

    if np.isnan(smape):
        return np.finfo(np.float64).max

    return smape


def smape(act, pred):
    pred = pred.astype(np.float64, copy=False)
    n = np.abs(pred - act)
    d = (np.abs(pred) + np.abs(act)) / 2
    ape = n / d
    smape = np.mean(ape)

    if np.isnan(smape):
        return np.finfo(np.float64).max

    return smape


def rmse(act, pred, weight=None):
    """
    RMSE = Root Mean Squared Error = sqrt( mean( (act - pred)**2 ) )
    """
    if len(pred.shape) > 1:
        if pred.shape[1] == 2:
            pred = pred[:, 1]
        else:
            pred = pred.ravel()

    pred = pred.astype(np.float64, copy=False)
    d = act - pred
    sd = np.power(d, 2)
    if weight is not None:
        if weight.sum() == 0:
            return 0
        sd = sd * weight / weight.mean()
    mse = sd.mean()
    rmse = np.sqrt(mse)

    if np.isnan(rmse):
        return np.finfo(np.float64).max
    else:
        return rmse


def gamma_loss(act, pred, weight=None):
    """Gamma deviance"""
    eps = 0.001
    pred = np.maximum(pred, eps)  # ensure predictions are strictly positive
    act = np.maximum(act, eps)  # ensure actuals are strictly positive
    d = 2 * (-np.log(act / pred) + (act - pred) / pred)
    if weight is not None:
        d = d * weight / np.mean(weight)
    return np.mean(d)


def tweedie_loss(act, pred, weight=None, p=1.5):
    """tweedie deviance for p = 1.5 only"""

    if p <= 1 or p >= 2:
        raise ValueError('p equal to %s is not supported' % p)

    eps = 0.001
    pred = np.maximum(pred, eps)  # ensure predictions are strictly positive
    act = np.maximum(act, 0)  # ensure actuals are not negative
    d = (
        (act ** (2.0 - p)) / ((1 - p) * (2 - p))
        - (act * (pred ** (1 - p))) / (1 - p)
        + (pred ** (2 - p)) / (2 - p)
    )
    d = 2 * d
    if weight is not None:
        d = d * weight / np.mean(weight)
    return np.mean(d)


def poisson_loss(act, pred, weight=None):
    """
        Poisson Deviance = 2*(act*log(act/pred)-(act-pred))
        ONLY WORKS FOR POSITIVE RESPONSES
    """
    if len(pred.shape) > 1:
        pred = pred.ravel()
    pred = np.maximum(pred, 1e-8)  # ensure predictions are strictly positive
    act = np.maximum(act, 0)  # ensure actuals are non-negative
    d = np.zeros(len(act))
    d[act == 0] = pred[act == 0]
    cond = act > 0
    d[cond] = act[cond] * np.log(act[cond] / pred[cond]) - (act[cond] - pred[cond])
    d = d * 2
    if weight is not None:
        if weight.sum() == 0:
            return 0
        d = d * weight / weight.mean()
    return d.mean()