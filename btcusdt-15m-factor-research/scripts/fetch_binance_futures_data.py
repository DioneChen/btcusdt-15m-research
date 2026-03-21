
import argparse
import json
from pathlib import Path

import requests

from btcusdt_15m_factor_research.data_pipeline import fetch_funding_rate, fetch_kline_like, to_ms


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', required=True)
    args = parser.parse_args()

    with open(args.config, 'r', encoding='utf-8') as f:
        cfg = json.load(f)

    workspace = Path(cfg['workspace_dir'])
    raw_dir = workspace / 'raw'
    raw_dir.mkdir(parents=True, exist_ok=True)

    session = requests.Session()
    session.headers.update({'User-Agent': 'Mozilla/5.0'})

    start_ms = to_ms(cfg['start'])
    end_ms = to_ms(cfg['end'])

    price_df = fetch_kline_like('/fapi/v1/klines', session, symbol=cfg['symbol'], interval=cfg['interval'], start_time=start_ms, end_time=end_ms)
    mark_df = fetch_kline_like('/fapi/v1/markPriceKlines', session, symbol=cfg['symbol'], interval=cfg['interval'], start_time=start_ms, end_time=end_ms)
    index_df = fetch_kline_like('/fapi/v1/indexPriceKlines', session, pair=cfg['pair'], interval=cfg['interval'], start_time=start_ms, end_time=end_ms)
    funding_df = fetch_funding_rate(session, cfg['symbol'], start_time=start_ms, end_time=end_ms)

    price_df.to_parquet(raw_dir / 'price_df.parquet', index=False)
    mark_df.to_parquet(raw_dir / 'mark_df.parquet', index=False)
    index_df.to_parquet(raw_dir / 'index_df.parquet', index=False)
    funding_df.to_parquet(raw_dir / 'funding_df.parquet', index=False)

    print('saved raw data to', raw_dir)


if __name__ == '__main__':
    main()
