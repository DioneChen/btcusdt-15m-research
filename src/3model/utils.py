import math
import random

import numpy as np
import pandas as pd
import torch
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score


def set_seed(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def rolling_zscore(s, window=14 * 24 * 4, min_periods=7 * 24 * 4):
    mu = s.rolling(window, min_periods=min_periods).mean()
    sd = s.rolling(window, min_periods=min_periods).std()
    z = (s - mu) / sd
    return z.replace([np.inf, -np.inf], np.nan)


def keep_closed_bars(df, end_time):
    out = df.copy()
    if end_time.tzinfo is None:
        end_time = end_time.tz_localize("UTC")
    else:
        end_time = end_time.tz_convert("UTC")
    out = out[out["open_time"] < end_time].copy()
    out = out.sort_values("open_time").drop_duplicates("open_time").reset_index(drop=True)
    return out


def time_split_mask(index, start, end):
    return (index >= start) & (index < end)


def regression_metrics(y_true, y_pred):
    y_true = np.asarray(y_true, dtype=float)
    y_pred = np.asarray(y_pred, dtype=float)

    rmse = math.sqrt(mean_squared_error(y_true, y_pred))
    mae = mean_absolute_error(y_true, y_pred)
    r2 = r2_score(y_true, y_pred)
    ic = pd.Series(y_true).corr(pd.Series(y_pred), method="pearson")
    rank_ic = pd.Series(y_true).corr(pd.Series(y_pred), method="spearman")
    direction_acc = float((np.sign(y_true) == np.sign(y_pred)).mean())

    return {
        "rmse": float(rmse),
        "mae": float(mae),
        "r2": float(r2),
        "ic": float(0.0 if pd.isna(ic) else ic),
        "rank_ic": float(0.0 if pd.isna(rank_ic) else rank_ic),
        "direction_acc": direction_acc,
    }


def max_drawdown_from_returns(rets):
    curve = (1 + rets.fillna(0)).cumprod()
    peak = curve.cummax()
    drawdown = curve / peak - 1
    return float(drawdown.min())
