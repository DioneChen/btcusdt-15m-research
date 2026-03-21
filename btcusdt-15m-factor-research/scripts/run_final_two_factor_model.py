
    import argparse
    import json
    from pathlib import Path

    import matplotlib.pyplot as plt
    import pandas as pd

    from btcusdt_15m_factor_research.backtest_utils import backtest_metrics, build_score, rebalance_4h_backtest


    FINAL_MODEL = {
        'vol_surge_4h_z_lag1': 1,
        'mom_8h_z_lag1': -1,
    }
    FINAL_THRESHOLD = {
        'name': 'q80_20',
        'long_th': 0.5210193934122136,
        'short_th': -0.574675299410354,
    }


    def main():
        parser = argparse.ArgumentParser()
        parser.add_argument('--config', required=True)
        args = parser.parse_args()

        with open(args.config, 'r', encoding='utf-8') as f:
            cfg = json.load(f)

        workspace = Path(cfg['workspace_dir'])
        checkpoint_dir = workspace / 'checkpoint'
        report_dir = workspace / 'final_two_factor_reports'
        report_dir.mkdir(parents=True, exist_ok=True)

        val = pd.read_parquet(checkpoint_dir / 'val.parquet')
        test = pd.read_parquet(checkpoint_dir / 'test.parquet')
        fee_bps = float(cfg.get('fee_bps', 0.0))
        ret_col = 'ret_15m'

        val_bt = rebalance_4h_backtest(build_score(val, FINAL_MODEL), 'score_raw', ret_col, FINAL_THRESHOLD['long_th'], FINAL_THRESHOLD['short_th'], fee_bps)
        test_bt = rebalance_4h_backtest(build_score(test, FINAL_MODEL), 'score_raw', ret_col, FINAL_THRESHOLD['long_th'], FINAL_THRESHOLD['short_th'], fee_bps)

        val_metrics = backtest_metrics(val_bt)
        test_metrics = backtest_metrics(test_bt)

        val_metrics.to_csv(report_dir / 'validation_metrics.csv')
        test_metrics.to_csv(report_dir / 'test_metrics.csv')
        val_bt.to_csv(report_dir / 'validation_backtest.csv')
        test_bt.to_csv(report_dir / 'test_backtest.csv')

        val_bt[['equity', 'bh_equity']].plot(figsize=(10, 4), title='Validation: Two-Factor Model (q80_20)')
        plt.tight_layout()
        plt.savefig(report_dir / 'validation_two_factor_q80_20.png', dpi=160)
        plt.close()

        test_bt[['equity', 'bh_equity']].plot(figsize=(10, 4), title='Final Test: Two-Factor Model')
        plt.tight_layout()
        plt.savefig(report_dir / 'test_two_factor.png', dpi=160)
        plt.close()

        print('validation metrics:')
        print(val_metrics)
        print('
test metrics:')
        print(test_metrics)


    if __name__ == '__main__':
        main()
