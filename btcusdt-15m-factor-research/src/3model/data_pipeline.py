import os

import numpy as np
import pandas as pd

from .config import BASE_FEATURE_COLS, TARGET_COL, RET_COL, TEST_END, TRAIN_END, TREE_LAGS
from .config import RAW_FILENAMES
from .utils import keep_closed_bars, rolling_zscore, time_split_mask


def _read_table_with_fallback(base_path_no_ext):
    parquet_path = base_path_no_ext + ".parquet"
    csv_path = base_path_no_ext + ".csv"
    if os.path.exists(parquet_path):
        try:
            return pd.read_parquet(parquet_path)
        except Exception:
            pass
    if os.path.exists(csv_path):
        return pd.read_csv(csv_path)
    raise FileNotFoundError("Neither parquet nor csv found for %s" % base_path_no_ext)


def load_raw_data(raw_dir):
    price_df = _read_table_with_fallback(os.path.join(raw_dir, RAW_FILENAMES["price"]))
    mark_df = _read_table_with_fallback(os.path.join(raw_dir, RAW_FILENAMES["mark"]))
    index_df = _read_table_with_fallback(os.path.join(raw_dir, RAW_FILENAMES["index"]))
    premium_df = _read_table_with_fallback(os.path.join(raw_dir, RAW_FILENAMES["premium"]))
    funding_df = _read_table_with_fallback(os.path.join(raw_dir, RAW_FILENAMES["funding"]))

    for df in [price_df, mark_df, index_df, premium_df]:
        df["open_time"] = pd.to_datetime(df["open_time"], utc=True)
    funding_df["fundingTime"] = pd.to_datetime(funding_df["fundingTime"], utc=True)

    numeric_cols_map = {
        "price": [
            "open", "high", "low", "close", "volume", "quote_asset_volume",
            "num_trades", "taker_buy_base_asset_volume", "taker_buy_quote_asset_volume",
        ],
        "mark": ["close"],
        "index": ["close"],
        "premium": ["close"],
        "funding": ["fundingRate", "markPrice"],
    }

    for col in numeric_cols_map["price"]:
        if col in price_df.columns:
            price_df[col] = pd.to_numeric(price_df[col], errors="coerce")
    for col in numeric_cols_map["mark"]:
        if col in mark_df.columns:
            mark_df[col] = pd.to_numeric(mark_df[col], errors="coerce")
    for col in numeric_cols_map["index"]:
        if col in index_df.columns:
            index_df[col] = pd.to_numeric(index_df[col], errors="coerce")
    for col in numeric_cols_map["premium"]:
        if col in premium_df.columns:
            premium_df[col] = pd.to_numeric(premium_df[col], errors="coerce")
    for col in numeric_cols_map["funding"]:
        if col in funding_df.columns:
            funding_df[col] = pd.to_numeric(funding_df[col], errors="coerce")

    return price_df, mark_df, index_df, premium_df, funding_df


def build_master_df(raw_dir):
    price_df, mark_df, index_df, premium_df, funding_df = load_raw_data(raw_dir)
    end_ts = TEST_END + pd.Timedelta(days=1)

    price_df_ = keep_closed_bars(price_df, end_ts)
    mark_df_ = keep_closed_bars(mark_df, end_ts)
    index_df_ = keep_closed_bars(index_df, end_ts)
    premium_df_ = keep_closed_bars(premium_df, end_ts)

    funding_df_ = funding_df[funding_df["fundingTime"] < end_ts].copy()
    funding_df_ = funding_df_.sort_values("fundingTime").drop_duplicates("fundingTime").reset_index(drop=True)

    price = price_df_[[
        "open_time", "open", "high", "low", "close",
        "volume", "quote_asset_volume", "num_trades",
        "taker_buy_base_asset_volume", "taker_buy_quote_asset_volume",
    ]].copy()
    mark = mark_df_[["open_time", "close"]].rename(columns={"close": "mark_close"})
    indexp = index_df_[["open_time", "close"]].rename(columns={"close": "index_close"})
    premium = premium_df_[["open_time", "close"]].rename(columns={"close": "premium_close"})
    funding = funding_df_[["fundingTime", "fundingRate"]].rename(
        columns={"fundingTime": "funding_time", "fundingRate": "funding_rate"}
    )

    df = price.merge(mark, on="open_time", how="left")
    df = df.merge(indexp, on="open_time", how="left")
    df = df.merge(premium, on="open_time", how="left")
    df = df.sort_values("open_time").reset_index(drop=True)

    funding = funding.sort_values("funding_time")
    df = pd.merge_asof(
        df.sort_values("open_time"),
        funding,
        left_on="open_time",
        right_on="funding_time",
        direction="backward",
    )

    df["funding_rate"] = df["funding_rate"].fillna(0.0)
    df = df.sort_values("open_time").drop_duplicates("open_time").reset_index(drop=True)
    return df


def build_feature_df(master_df):
    df = master_df.copy().sort_values("open_time").reset_index(drop=True)

    # base returns
    df[RET_COL] = df["close"].pct_change()
    df["ret_1h"] = df["close"].pct_change(4)
    df["ret_4h"] = df["close"].pct_change(16)
    df["ret_8h"] = df["close"].pct_change(32)

    # target: next 4h return
    df[TARGET_COL] = df["close"].shift(-16) / df["close"] - 1

    # realized volatility
    df["rv_4h"] = df[RET_COL].rolling(16).std()
    df["rv_8h"] = df[RET_COL].rolling(32).std()

    # reversal / momentum
    df["rev_1h"] = -df["ret_1h"]
    df["rev_4h"] = -df["ret_4h"]
    df["mom_4h"] = df["ret_4h"]
    df["mom_8h"] = df["ret_8h"]

    # volume features
    df["vol_4h"] = df["volume"].rolling(16).mean()
    df["vol_24h"] = df["volume"].rolling(96).mean()
    df["vol_surge_4h"] = df["volume"] / df["vol_4h"].replace(0, np.nan)

    # basis and spread
    df["mark_index_spread"] = (df["mark_close"] - df["index_close"]) / df["index_close"].replace(0, np.nan)
    df["spread_change_4h"] = df["mark_index_spread"].diff(16)

    # order-flow imbalance
    df["taker_buy_imbalance"] = (
        2.0 * df["taker_buy_base_asset_volume"] / df["volume"].replace(0, np.nan) - 1.0
    )

    # funding features
    df["funding_state"] = df["funding_rate"]
    df["funding_change"] = df["funding_rate"].diff(1)
    df["funding_dev"] = df["funding_rate"] - df["funding_rate"].rolling(96).mean()

    for base_col in [
        "rv_4h", "rv_8h", "rev_1h", "rev_4h", "mom_4h", "mom_8h", "vol_4h", "vol_24h",
        "vol_surge_4h", "mark_index_spread", "taker_buy_imbalance", "spread_change_4h",
        "funding_state", "funding_change", "funding_dev", RET_COL,
    ]:
        df[base_col + "_z"] = rolling_zscore(df[base_col])

    use_cols = ["open_time", TARGET_COL] + BASE_FEATURE_COLS
    df_model = df[use_cols].dropna().copy()
    df_model = df_model.set_index("open_time").sort_index()
    return df_model


def add_tree_lag_features(df_model, feature_cols, lags):
    df = df_model.copy()
    for col in feature_cols:
        for lag in lags:
            df["%s_lag%s" % (col, lag)] = df[col].shift(lag)
    keep_cols = [c for c in df.columns if c.endswith(tuple(["lag%s" % l for l in lags]))] + [TARGET_COL]
    df = df[keep_cols].dropna().copy()
    return df
