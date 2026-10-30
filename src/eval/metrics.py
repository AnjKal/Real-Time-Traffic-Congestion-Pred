"""Masked MAE / RMSE / MAPE for arrays shaped (N, horizon, nodes).

Cells where the ground truth is 0 or NaN are treated as missing and ignored. `where` is an
optional extra boolean mask broadcastable to (N, horizon, nodes) for slicing (e.g. rush hour).
"""
import numpy as np


def _mask(y_true, where=None):
    m = np.isfinite(y_true) & (y_true != 0)
    return m if where is None else m & where


def _masked_mean(values, mask, axis=None):
    values = np.where(mask, values, 0.0)
    count = mask.sum(axis=axis)
    with np.errstate(invalid="ignore", divide="ignore"):
        return values.sum(axis=axis) / count        # NaN where nothing is unmasked


def _mae(y_true, y_pred, where, axis):
    return _masked_mean(np.abs(y_pred - y_true), _mask(y_true, where), axis)


def _rmse(y_true, y_pred, where, axis):
    return np.sqrt(_masked_mean((y_pred - y_true) ** 2, _mask(y_true, where), axis))


def _mape(y_true, y_pred, where, axis):
    safe = np.where(y_true == 0, 1.0, y_true)
    return 100.0 * _masked_mean(np.abs((y_pred - y_true) / safe), _mask(y_true, where), axis)


def masked_mae(y_true, y_pred, where=None):
    return float(_mae(y_true, y_pred, where, None))


def masked_rmse(y_true, y_pred, where=None):
    return float(_rmse(y_true, y_pred, where, None))


def masked_mape(y_true, y_pred, where=None):
    return float(_mape(y_true, y_pred, where, None))


def masked_mae_per_horizon(y_true, y_pred, where=None):
    return _mae(y_true, y_pred, where, (0, 2))


def masked_rmse_per_horizon(y_true, y_pred, where=None):
    return _rmse(y_true, y_pred, where, (0, 2))


def masked_mape_per_horizon(y_true, y_pred, where=None):
    return _mape(y_true, y_pred, where, (0, 2))


def masked_mae_per_node(y_true, y_pred, where=None):
    return _mae(y_true, y_pred, where, (0, 1))


def masked_rmse_per_node(y_true, y_pred, where=None):
    return _rmse(y_true, y_pred, where, (0, 1))


def masked_mape_per_node(y_true, y_pred, where=None):
    return _mape(y_true, y_pred, where, (0, 1))
