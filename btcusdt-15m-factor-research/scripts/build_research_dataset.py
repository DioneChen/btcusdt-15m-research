
import argparse
import json
from pathlib import Path

import pandas as pd

from btcusdt_15m_factor_research.feature_engineering import add_features, build_master_dataframe, keep_closed_bars


def load_raw_parquet(raw_dir: Path, name: str) -> pd.DataFrame:
    return pd.read_parquet(raw_dir / f'{name}.parquet')


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', required=True)
    args = parser.parse_args()

    with open(args.config, 'r', encoding='utf-8') as f:
        cfg = json.load(f)

    workspace = Path(cfg['workspace_dir'])
    raw_dir = workspace / 'raw'
    checkpoint_dir = workspace / 'checkpoint'
    checkpoint_dir.mkdir(parents=True, exist_ok=True)

    price_df = load_raw_parquet(raw_dir, 'price_df')
    mark_df = load_raw_parquet(raw_dir, 'mark_df')
    index_df = load_raw_parquet(raw_dir, 'index_df')
    funding_df = load_raw_parquet(raw_dir, 'funding_df')

    end_ts = pd.Timestamp(cfg['end'], tz='UTC')
    price_df = keep_closed_bars(price_df, end_ts)
    mark_df = keep_closed_bars(mark_df, end_ts)
    index_df = keep_closed_bars(index_df, end_ts)

    master_df = build_master_dataframe(price_df, mark_df, index_df, funding_df)
    master_df, factor_cols, z_factors, z_factors_lag1 = add_features(master_df)

    research_cols = ['fwd_ret_4h', 'fwd_log_ret_4h', 'ret_15m', 'log_ret_15m'] + factor_cols + z_factors + z_factors_lag1
    df_model = master_df[research_cols].copy()
    df_model = df_model[df_model['fwd_ret_4h'].notna()].copy()

    train = df_model[(df_model.index >= pd.Timestamp(cfg['train_start'], tz='UTC')) & (df_model.index < pd.Timestamp(cfg['train_end'], tz='UTC'))].copy()
    val = df_model[(df_model.index >= pd.Timestamp(cfg['val_start'], tz='UTC')) & (df_model.index < pd.Timestamp(cfg['val_end'], tz='UTC'))].copy()
    test = df_model[(df_model.index >= pd.Timestamp(cfg['test_start'], tz='UTC')) & (df_model.index < pd.Timestamp(cfg['test_end'], tz='UTC'))].copy()

    master_df.to_parquet(checkpoint_dir / 'master_df.parquet')
    df_model.to_parquet(checkpoint_dir / 'df_model.parquet')
    train.to_parquet(checkpoint_dir / 'train.parquet')
    val.to_parquet(checkpoint_dir / 'val.parquet')
    test.to_parquet(checkpoint_dir / 'test.parquet')

    meta = {
        'symbol': cfg['symbol'],
        'pair': cfg['pair'],
        'interval': cfg['interval'],
        'start': cfg['start'],
        'end': cfg['end'],
        'factor_cols': factor_cols,
        'z_factors': z_factors,
        'z_factors_lag1': z_factors_lag1,
        'target': 'fwd_ret_4h',
        'ret_col': 'ret_15m',
        'train_shape': list(train.shape),
        'val_shape': list(val.shape),
        'test_shape': list(test.shape),
    }
    with open(checkpoint_dir / 'meta.json', 'w', encoding='utf-8') as f:
        json.dump(meta, f, ensure_ascii=False, indent=2)

    print('checkpoint saved to', checkpoint_dir)


if __name__ == '__main__':
    main()
