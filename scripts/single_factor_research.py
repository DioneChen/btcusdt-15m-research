
import argparse
import json
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd

from btcusdt_15m_factor_research.research_utils import factor_ic_table, monthly_rank_ic_series, monthly_rank_ic_summary, quantile_test


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', required=True)
    args = parser.parse_args()

    with open(args.config, 'r', encoding='utf-8') as f:
        cfg = json.load(f)

    workspace = Path(cfg['workspace_dir'])
    checkpoint_dir = workspace / 'checkpoint'
    report_dir = workspace / 'single_factor_reports'
    report_dir.mkdir(parents=True, exist_ok=True)

    train = pd.read_parquet(checkpoint_dir / 'train.parquet')
    with open(checkpoint_dir / 'meta.json', 'r', encoding='utf-8') as f:
        meta = json.load(f)

    use_factors = meta['z_factors_lag1']
    target = meta['target']

    train_ic = factor_ic_table(train, use_factors, target)
    train_monthly = monthly_rank_ic_summary(train, use_factors, target)
    factor_summary = train_ic.merge(train_monthly, on='factor', how='left')
    factor_summary.to_csv(report_dir / 'factor_summary.csv', index=False)

    for factor in use_factors:
        qt = quantile_test(train, factor, target, q=5)
        qt.to_csv(report_dir / f'quantile_{factor}.csv')
        s = monthly_rank_ic_series(train, factor, target).dropna()
        if len(s) > 0:
            plt.figure(figsize=(10, 4))
            s.plot(title=f'Train Monthly RankIC: {factor}')
            plt.axhline(0, linestyle='--')
            plt.tight_layout()
            plt.savefig(report_dir / f'monthly_rankic_{factor}.png', dpi=150)
            plt.close()

    print('single-factor reports saved to', report_dir)


if __name__ == '__main__':
    main()
