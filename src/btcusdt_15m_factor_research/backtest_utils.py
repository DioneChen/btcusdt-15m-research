
from typing import Dict

import numpy as np
import pandas as pd


def build_score(data: pd.DataFrame, factor_signs: Dict[str, int]) -> pd.DataFrame:
    out = data.copy()
    weight = 1.0 / len(factor_signs)
    out["score_raw"] = 0.0
    for factor, sign in factor_signs.items():
        out["score_raw"] += weight * sign * out[factor]
    return out


def rebalance_4h_backtest(data: pd.DataFrame, score_col: str, ret_col: str, long_th: float, short_th: float, fee_bps: float) -> pd.DataFrame:
    tmp = data[[score_col, ret_col]].dropna().copy()
    is_rebalance = (tmp.index.hour % 4 == 0) & (tmp.index.minute == 0)

    tmp["signal"] = np.nan
    tmp.loc[is_rebalance & (tmp[score_col] > long_th), "signal"] = 1
    tmp.loc[is_rebalance & (tmp[score_col] < short_th), "signal"] = -1
    tmp.loc[is_rebalance & (tmp[score_col].between(short_th, long_th)), "signal"] = 0

    tmp["position"] = tmp["signal"].ffill().fillna(0)
    tmp["position_lag"] = tmp["position"].shift(1).fillna(0)
    tmp["turnover"] = (tmp["position"] - tmp["position"].shift(1).fillna(0)).abs()

    fee_rate = fee_bps / 10000.0
    tmp["strategy_ret"] = tmp["position_lag"] * tmp[ret_col] - tmp["turnover"] * fee_rate
    tmp["equity"] = (1 + tmp["strategy_ret"]).cumprod()
    tmp["bh_equity"] = (1 + tmp[ret_col].fillna(0)).cumprod()
    return tmp


def backtest_metrics(bt: pd.DataFrame, ret_col: str = "strategy_ret") -> pd.Series:
    s = bt[ret_col].dropna()
    if len(s) == 0:
        return pd.Series(dtype=float)

    equity = (1 + s).cumprod()
    total_return = equity.iloc[-1] - 1
    elapsed_hours = (bt.index[-1] - bt.index[0]).total_seconds() / 3600.0
    years = elapsed_hours / (24 * 365)
    annual_return = equity.iloc[-1] ** (1 / years) - 1 if years > 0 else np.nan
    annual_vol = s.std() * np.sqrt(24 * 4 * 365)
    sharpe = (s.mean() / s.std()) * np.sqrt(24 * 4 * 365) if s.std() > 0 else np.nan
    peak = equity.cummax()
    drawdown = equity / peak - 1
    return pd.Series({
        "total_return": total_return,
        "annual_return": annual_return,
        "annual_vol": annual_vol,
        "sharpe": sharpe,
        "max_drawdown": drawdown.min(),
        "win_rate": (s > 0).mean(),
        "avg_turnover": bt["turnover"].mean(),
        "num_obs": len(s),
    })
