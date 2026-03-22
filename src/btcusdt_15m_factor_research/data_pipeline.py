
import time
from typing import Optional

import pandas as pd
import requests


BASE_URL = "https://fapi.binance.com"


def to_ms(dt_str: str) -> int:
    return int(pd.Timestamp(dt_str, tz="UTC").timestamp() * 1000)


def fetch_kline_like(
    endpoint: str,
    session: requests.Session,
    symbol: Optional[str] = None,
    pair: Optional[str] = None,
    interval: str = "15m",
    start_time: Optional[int] = None,
    end_time: Optional[int] = None,
    limit: int = 1500,
) -> pd.DataFrame:
    rows = []
    current = start_time

    while True:
        params = {"interval": interval, "limit": limit}
        if symbol is not None:
            params["symbol"] = symbol
        if pair is not None:
            params["pair"] = pair
        if current is not None:
            params["startTime"] = current
        if end_time is not None:
            params["endTime"] = end_time

        resp = session.get(BASE_URL + endpoint, params=params, timeout=30)
        resp.raise_for_status()
        batch = resp.json()
        if not batch:
            break
        rows.extend(batch)

        last_open_time = batch[-1][0]
        current = last_open_time + 1
        if len(batch) < limit:
            break
        if end_time is not None and last_open_time >= end_time:
            break
        time.sleep(0.1)

    cols = [
        "open_time", "open", "high", "low", "close", "volume", "close_time",
        "quote_asset_volume", "num_trades", "taker_buy_base_asset_volume",
        "taker_buy_quote_asset_volume", "ignore",
    ]
    df = pd.DataFrame(rows, columns=cols)
    for c in ["open", "high", "low", "close", "volume", "quote_asset_volume",
              "taker_buy_base_asset_volume", "taker_buy_quote_asset_volume"]:
        df[c] = pd.to_numeric(df[c], errors="coerce")
    df["open_time"] = pd.to_datetime(df["open_time"], unit="ms", utc=True)
    df["close_time"] = pd.to_datetime(df["close_time"], unit="ms", utc=True)
    return df


def fetch_funding_rate(
    session: requests.Session,
    symbol: str,
    start_time: Optional[int] = None,
    end_time: Optional[int] = None,
    limit: int = 1000,
) -> pd.DataFrame:
    rows = []
    current = start_time
    while True:
        params = {"symbol": symbol, "limit": limit}
        if current is not None:
            params["startTime"] = current
        if end_time is not None:
            params["endTime"] = end_time
        resp = session.get(BASE_URL + "/fapi/v1/fundingRate", params=params, timeout=30)
        resp.raise_for_status()
        batch = resp.json()
        if not batch:
            break
        rows.extend(batch)
        last_ts = int(batch[-1]["fundingTime"])
        current = last_ts + 1
        if len(batch) < limit:
            break
        if end_time is not None and last_ts >= end_time:
            break
        time.sleep(0.1)

    df = pd.DataFrame(rows)
    if len(df) == 0:
        return df
    df["fundingTime"] = pd.to_datetime(df["fundingTime"], unit="ms", utc=True)
    for c in ["fundingRate", "markPrice"]:
        df[c] = pd.to_numeric(df[c], errors="coerce")
    return df
