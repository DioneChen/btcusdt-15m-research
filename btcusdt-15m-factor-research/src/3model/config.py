import os
import pandas as pd

SYMBOL = "BTCUSDT"
INTERVAL = "15m"
TARGET_COL = "fwd_ret_4h"
RET_COL = "ret_15m"
HOLD_BARS = 16            # 4h = 16 x 15m
SEQ_LEN = 32              # LSTM looks back 8 hours
FEE_BPS = 5.0             # one-way trading fee in basis points
SEED = 42

TRAIN_START = pd.Timestamp("2025-06-01", tz="UTC")
TRAIN_END = pd.Timestamp("2025-12-01", tz="UTC")
VAL_START = pd.Timestamp("2025-12-01", tz="UTC")
VAL_END = pd.Timestamp("2026-01-16", tz="UTC")
TEST_START = pd.Timestamp("2026-01-16", tz="UTC")
TEST_END = pd.Timestamp("2026-03-01", tz="UTC")

BASE_FEATURE_COLS = [
    "rv_4h_z",
    "rv_8h_z",
    "rev_1h_z",
    "rev_4h_z",
    "mom_4h_z",
    "mom_8h_z",
    "vol_4h_z",
    "vol_24h_z",
    "vol_surge_4h_z",
    "mark_index_spread_z",
    "taker_buy_imbalance_z",
    "spread_change_4h_z",
    "funding_state_z",
    "funding_change_z",
    "funding_dev_z",
    "ret_15m_z",
]
TREE_LAGS = [0, 1, 4, 16]

RAW_FILENAMES = {
    "price": "price_df",
    "mark": "mark_df",
    "index": "index_df",
    "premium": "premium_df",
    "funding": "funding_df",
}


def ensure_dir(path):
    os.makedirs(path, exist_ok=True)
