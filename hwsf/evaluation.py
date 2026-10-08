"""Evaluation metrics (Section III-D): RMSE, MAE, bias, R^2, and the MSE decomposition of equation (3).

All metrics weight every municipality-year record equally; bias means prediction minus observation.
"""
import numpy as np
import pandas as pd


def metrics(y_true, y_pred):
    """RMSE, MAE, bias, and R^2 (kg/10a)."""
    y_true, y_pred = np.asarray(y_true, float), np.asarray(y_pred, float)
    e = y_pred - y_true
    return dict(n=len(e), rmse=float(np.sqrt(np.mean(e ** 2))), mae=float(np.mean(np.abs(e))), bias=float(np.mean(e)),
                r2=float(1 - np.sum(e ** 2) / np.sum((y_true - y_true.mean()) ** 2)))


def mse_decomposition(frame, prediction="estimate", truth="y_true"):
    """Equation (3): MSE = (1/N) sum_ct n_c ebar_ct^2 + (1/N) sum_ct sum_i (e_ict - ebar_ct)^2.

    ``frame`` needs the columns prefecture, year, and the prediction and observation columns. Returns the
    prefecture-year term and the centered municipal term."""
    e = frame[prediction] - frame[truth]
    ebar = e.groupby([frame.prefecture, frame.year]).transform("mean")
    n = len(e)
    return dict(prefecture_year=float((ebar ** 2).sum() / n), municipal=float(((e - ebar) ** 2).sum() / n))


def summary_table(frame, prediction="estimate", truth="y_true", by=None):
    """Metrics overall or for each group of ``by`` (e.g. "year" or "prefecture")."""
    if by is None:
        return pd.DataFrame([metrics(frame[truth], frame[prediction])])
    rows = [dict({by: key}, **metrics(g[truth], g[prediction])) for key, g in frame.groupby(by)]
    return pd.DataFrame(rows)


def interval_summary(intervals, by=None):
    """Coverage, mean width, and mean interval score of ``uncertainty.forward_intervals`` output."""
    def one(g):
        return dict(n=len(g), hits=int(g.hit.sum()), coverage=float(g.hit.mean()), mean_width=float(g.width.mean()),
                    mean_interval_score=float(g.interval_score.mean()))
    if by is None:
        return pd.DataFrame([one(intervals)])
    return pd.DataFrame([dict({by: k}, **one(g)) for k, g in intervals.groupby(by)])
