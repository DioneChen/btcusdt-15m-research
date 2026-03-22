
import argparse
import json
from pathlib import Path

import pandas as pd

from btcusdt_15m_factor_research.backtest_utils import backtest_metrics, build_score, rebalance_4h_backtest


CANDIDATE_MODELS = {
    'A_core3': {
        'vol_surge_4h_z_lag1': 1,
        'vol_24h_z_lag1': -1,
        'mom_8h_z_lag1': -1,
    },
    'B_core4': {
        'vol_surge_4h_z_lag1': 1,
        'vol_24h_z_lag1': -1,
        'mom_8h_z_lag1': -1,
        'mom_4h_z_lag1': 1,
    },
    'E_core5_funding': {
        'vol_surge_4h_z_lag1': 1,
        'vol_24h_z_lag1': -1,
        'mom_8h_z_lag1': -1,
        'mom_4h_z_lag1': 1,
        'funding_dev_24h_z_lag1': 1,
    },
    'two_factor_volsurge_mom8h': {
        'vol_surge_4h_z_lag1': 1,
        'mom_8h_z_lag1': -1,
    },
}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', required=True)
    args = parser.parse_args()

    with open(args.config, 'r', encoding='utf-8') as f:
        cfg = json.load(f)

    workspace = Path(cfg['workspace_dir'])
    checkpoint_dir = workspace / 'checkpoint'
    report_dir = workspace / 'validation_reports'
    report_dir.mkdir(parents=True, exist_ok=True)

    train = pd.read_parquet(checkpoint_dir / 'train.parquet')
    val = pd.read_parquet(checkpoint_dir / 'val.parquet')

    fee_bps = float(cfg.get('fee_bps', 0.0))
    ret_col = 'ret_15m'

    rows = []
    for model_name, factor_signs in CANDIDATE_MODELS.items():
        train_scored = build_score(train, factor_signs)
        val_scored = build_score(val, factor_signs)
        train_rebalance_scores = train_scored.loc[(train_scored.index.hour % 4 == 0) & (train_scored.index.minute == 0), 'score_raw'].dropna()
        threshold_sets = {
            'q60_40': (train_rebalance_scores.quantile(0.60), train_rebalance_scores.quantile(0.40)),
            'q70_30': (train_rebalance_scores.quantile(0.70), train_rebalance_scores.quantile(0.30)),
            'q80_20': (train_rebalance_scores.quantile(0.80), train_rebalance_scores.quantile(0.20)),
        }
        for threshold_name, (long_th, short_th) in threshold_sets.items():
            bt = rebalance_4h_backtest(val_scored, 'score_raw', ret_col, float(long_th), float(short_th), fee_bps)
            metrics = backtest_metrics(bt)
            row = metrics.to_dict()
            row.update({
                'model': model_name,
                'threshold': threshold_name,
                'long_th': float(long_th),
                'short_th': float(short_th),
            })
            rows.append(row)
    out = pd.DataFrame(rows).sort_values(['sharpe', 'total_return'], ascending=False).reset_index(drop=True)
    out.to_csv(report_dir / 'validation_model_comparison.csv', index=False)
    print('validation results saved to', report_dir / 'validation_model_comparison.csv')


if __name__ == '__main__':
    main()
