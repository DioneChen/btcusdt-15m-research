
from typing import List, Tuple

import numpy as np
import pandas as pd


def keep_closed_bars(df: pd.DataFrame, end_time: pd.Timestamp) -> pd.DataFrame:
    out = df.copy()
    out = out[out["close_time"] <= end_time].copy()
    return out.reset_index(drop=True)


def rolling_zscore(s: pd.Series, window: int = 14 * 24 * 4, min_periods: int = 7 * 24 * 4) -> pd.Series:
    mu = s.rolling(window, min_periods=min_periods).mean()
    sd = s.rolling(window, min_periods=min_periods).std()
    sd = sd.mask(sd < 1e-12, np.nan)
    z = (s - mu) / sd
    return z.replace([np.inf, -np.inf], np.nan)


def build_master_dataframe(price_df: pd.DataFrame, mark_df: pd.DataFrame, index_df: pd.DataFrame, funding_df: pd.DataFrame) -> pd.DataFrame:
    price = price_df[[
        "open_time", "open", "high", "low", "close", "volume", "quote_asset_volume",
        "num_trades", "taker_buy_base_asset_volume",
    ]].rename(columns={"open_time": "open_time"})

    mark = mark_df[["open_time", "close"]].rename(columns={"close": "mark_close"})
    index = index_df[["open_time", "close"]].rename(columns={"close": "index_close"})
    funding = funding_df[["fundingTime", "fundingRate", "markPrice"]].rename(
        columns={"fundingTime": "funding_time", "fundingRate": "funding_rate", "markPrice": "funding_mark_price"}
    )

    df = price.merge(mark, on="open_time", how="left").merge(index, on="open_time", how="left")
    if len(funding) > 0:
        df = pd.merge_asof(
            df.sort_values("open_time"),
            funding.sort_values("funding_time"),
            left_on="open_time",
            right_on="funding_time",
            direction="backward",
        )
    return df.set_index("open_time").sort_index()


def add_features(df: pd.DataFrame) -> Tuple[pd.DataFrame, List[str], List[str], List[str]]:
    out = df.copy()
    out["ret_15m"] = out["close"].pct_change()
    out["log_ret_15m"] = np.log(out["close"]).diff()
    out["fwd_ret_4h"] = out["close"].shift(-16) / out["close"] - 1
    out["fwd_log_ret_4h"] = np.log(out["close"].shift(-16)) - np.log(out["close"])

    out["rv_4h"] = out["log_ret_15m"].rolling(16, min_periods=16).std()
    out["rv_8h"] = out["log_ret_15m"].rolling(32, min_periods=32).std()
    out["rev_1h"] = -(out["close"] / out["close"].shift(4) - 1)
    out["mom_4h"] = out["close"] / out["close"].shift(16) - 1
    out["mom_8h"] = out["close"] / out["close"].shift(32) - 1
    out["vol_4h"] = out["volume"].rolling(16, min_periods=16).sum()
    out["vol_24h"] = out["volume"].rolling(96, min_periods=96).mean()
    out["vol_surge_4h"] = out["vol_4h"] / (out["vol_24h"] * 16)
    out["mark_index_spread"] = out["mark_close"] / out["index_close"] - 1

    safe_volume = out["volume"].where(out["volume"] > 0, np.nan)
    out["taker_buy_ratio"] = out["taker_buy_base_asset_volume"] / safe_volume
    out["taker_buy_imbalance"] = out["taker_buy_ratio"] - 0.5
    out["spread_change_4h"] = out["mark_index_spread"] - out["mark_index_spread"].shift(16)
    out["funding_state"] = out["funding_rate"]
    out["funding_change_8h"] = out["funding_rate"] - out["funding_rate"].shift(32)
    out["funding_dev_24h"] = out["funding_rate"] - out["funding_rate"].rolling(96, min_periods=32).mean()

    factor_cols = [
        "rv_4h", "rv_8h", "rev_1h", "mom_4h", "mom_8h", "vol_4h", "vol_24h", "vol_surge_4h",
        "mark_index_spread", "taker_buy_imbalance", "spread_change_4h", "funding_state",
        "funding_change_8h", "funding_dev_24h",
    ]

    for c in factor_cols:
        out[c] = out[c].replace([np.inf, -np.inf], np.nan)

    z_factors = []
    z_factors_lag1 = []
    for c in factor_cols:
        z = c + "_z"
        z_lag1 = c + "_z_lag1"
        out[z] = rolling_zscore(out[c])
        out[z_lag1] = out[z].shift(1)
        z_factors.append(z)
        z_factors_lag1.append(z_lag1)

    return out, factor_cols, z_factors, z_factors_lag1
