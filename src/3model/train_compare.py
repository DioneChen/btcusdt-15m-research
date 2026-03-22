import argparse
import json
import os

import pandas as pd

from .config import (
    BASE_FEATURE_COLS,
    HOLD_BARS,
    INTERVAL,
    SEED,
    SEQ_LEN,
    SYMBOL,
    TARGET_COL,
    TEST_END,
    TEST_START,
    TRAIN_END,
    TRAIN_START,
    TREE_LAGS,
    VAL_END,
    VAL_START,
    ensure_dir,
)
from .data_pipeline import add_tree_lag_features, build_feature_df, build_master_df
from .evaluation import evaluate_one_model
from .models import build_lstm_arrays, predict_lstm, train_lgb, train_lstm, train_xgb
from .utils import set_seed, time_split_mask


def run_pipeline(raw_dir, output_dir):
    ensure_dir(output_dir)
    set_seed(SEED)

    print("=== 1) build features ===")
    master_df = build_master_df(raw_dir)
    df_model = build_feature_df(master_df)
    try:
        df_model.to_parquet(os.path.join(output_dir, "df_model_final.parquet"))
    except Exception:
        df_model.reset_index().to_csv(os.path.join(output_dir, "df_model_final.csv"), index=False)
    print("df_model shape:", df_model.shape)

    print("\n=== 2) tabular features ===")
    df_tree = add_tree_lag_features(df_model, BASE_FEATURE_COLS, TREE_LAGS)
    feature_cols_tree = [c for c in df_tree.columns if c != TARGET_COL]

    train_mask_tree = time_split_mask(df_tree.index, TRAIN_START, TRAIN_END)
    val_mask_tree = time_split_mask(df_tree.index, VAL_START, VAL_END)
    test_mask_tree = time_split_mask(df_tree.index, TEST_START, TEST_END)

    X_train = df_tree.loc[train_mask_tree, feature_cols_tree]
    y_train = df_tree.loc[train_mask_tree, TARGET_COL]
    X_val = df_tree.loc[val_mask_tree, feature_cols_tree]
    y_val = df_tree.loc[val_mask_tree, TARGET_COL]
    X_test = df_tree.loc[test_mask_tree, feature_cols_tree]
    y_test = df_tree.loc[test_mask_tree, TARGET_COL]

    print("tree train/val/test:", X_train.shape, X_val.shape, X_test.shape)
    results = []

    print("\n=== 3) train XGBoost ===")
    xgb_model = train_xgb(X_train, y_train, X_val, y_val)
    xgb_pred_df = pd.concat([
        pd.DataFrame({"ts": X_val.index, TARGET_COL: y_val.values, "pred": xgb_model.predict(X_val), "split": "val"}),
        pd.DataFrame({"ts": X_test.index, TARGET_COL: y_test.values, "pred": xgb_model.predict(X_test), "split": "test"}),
    ], ignore_index=True)
    results.append(evaluate_one_model("xgboost", xgb_pred_df, output_dir))

    print("\n=== 4) train LightGBM ===")
    lgb_model = train_lgb(X_train, y_train, X_val, y_val)
    lgb_pred_df = pd.concat([
        pd.DataFrame({"ts": X_val.index, TARGET_COL: y_val.values, "pred": lgb_model.predict(X_val), "split": "val"}),
        pd.DataFrame({"ts": X_test.index, TARGET_COL: y_test.values, "pred": lgb_model.predict(X_test), "split": "test"}),
    ], ignore_index=True)
    results.append(evaluate_one_model("lightgbm", lgb_pred_df, output_dir))

    print("\n=== 5) train LSTM ===")
    X_seq, y_seq, idx_seq = build_lstm_arrays(df_model, BASE_FEATURE_COLS, SEQ_LEN)
    train_mask_seq = time_split_mask(idx_seq, TRAIN_START, TRAIN_END)
    val_mask_seq = time_split_mask(idx_seq, VAL_START, VAL_END)
    test_mask_seq = time_split_mask(idx_seq, TEST_START, TEST_END)

    X_train_seq, y_train_seq = X_seq[train_mask_seq], y_seq[train_mask_seq]
    X_val_seq, y_val_seq = X_seq[val_mask_seq], y_seq[val_mask_seq]
    X_test_seq, y_test_seq = X_seq[test_mask_seq], y_seq[test_mask_seq]

    print("lstm train/val/test:", X_train_seq.shape, X_val_seq.shape, X_test_seq.shape)
    lstm_model, lstm_history = train_lstm(X_train_seq, y_train_seq, X_val_seq, y_val_seq, input_size=len(BASE_FEATURE_COLS))
    lstm_history.to_csv(os.path.join(output_dir, "lstm_training_history.csv"), index=False)

    lstm_pred_df = pd.concat([
        pd.DataFrame({"ts": idx_seq[val_mask_seq], TARGET_COL: y_val_seq, "pred": predict_lstm(lstm_model, X_val_seq), "split": "val"}),
        pd.DataFrame({"ts": idx_seq[test_mask_seq], TARGET_COL: y_test_seq, "pred": predict_lstm(lstm_model, X_test_seq), "split": "test"}),
    ], ignore_index=True)
    results.append(evaluate_one_model("lstm", lstm_pred_df, output_dir))

    print("\n=== 6) compare ===")
    result_df = pd.DataFrame(results).set_index("model")
    result_df = result_df.sort_values(["test_ic", "test_sharpe"], ascending=False)
    result_df.to_csv(os.path.join(output_dir, "model_compare_summary.csv"))
    print(result_df[[
        "val_ic", "test_ic", "val_rank_ic", "test_rank_ic",
        "val_direction_acc", "test_direction_acc",
        "val_rmse", "test_rmse",
        "val_sharpe", "test_sharpe",
        "val_total_return", "test_total_return",
        "test_max_drawdown",
    ]])

    best_model = result_df.index[0]
    print("\nBest model:", best_model)

    meta = {
        "symbol": SYMBOL,
        "interval": INTERVAL,
        "target": TARGET_COL,
        "hold_bars": HOLD_BARS,
        "seq_len": SEQ_LEN,
        "base_feature_cols": BASE_FEATURE_COLS,
        "tree_lags": TREE_LAGS,
    }
    with open(os.path.join(output_dir, "run_meta.json"), "w", encoding="utf-8") as f:
        json.dump(meta, f, ensure_ascii=False, indent=2)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--raw-dir", default="data/raw")
    parser.add_argument("--output-dir", default="outputs")
    args = parser.parse_args()
    run_pipeline(args.raw_dir, args.output_dir)


if __name__ == "__main__":
    main()
