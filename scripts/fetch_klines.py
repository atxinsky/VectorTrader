"""
拉取 Binance 历史 K线数据，存为 CSV。
用法: python scripts/fetch_klines.py [--symbol BTCUSDT] [--interval 1h] [--start 2020-01-01]
"""
import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import argparse
import pandas as pd
from datetime import datetime
from binance.client import Client
from config.settings import DATA_DIR, BINANCE_API_KEY, BINANCE_API_SECRET


def fetch_klines(symbol: str, interval: str, start: str, end: str | None = None):
    """从 Binance 拉取历史K线，自动分页"""
    client = Client(BINANCE_API_KEY, BINANCE_API_SECRET)

    print(f"拉取 {symbol} {interval} K线数据，从 {start} 开始...")
    klines = client.get_historical_klines(
        symbol=symbol,
        interval=interval,
        start_str=start,
        end_str=end,
    )
    print(f"共获取 {len(klines)} 根K线")

    df = pd.DataFrame(klines, columns=[
        'open_time', 'open', 'high', 'low', 'close', 'volume',
        'close_time', 'quote_volume', 'trades',
        'taker_buy_base', 'taker_buy_quote', 'ignore'
    ])

    # 类型转换
    for col in ['open', 'high', 'low', 'close', 'volume', 'quote_volume']:
        df[col] = pd.to_numeric(df[col], errors='coerce')

    df['open_time'] = pd.to_datetime(df['open_time'], unit='ms')
    df['close_time'] = pd.to_datetime(df['close_time'], unit='ms')

    # 只保留需要的列
    df = df[['open_time', 'open', 'high', 'low', 'close', 'volume']].copy()
    df.set_index('open_time', inplace=True)
    df.index.name = 'datetime'

    return df


def main():
    parser = argparse.ArgumentParser(description='Fetch Binance kline data')
    parser.add_argument('--symbol', default='BTCUSDT', help='交易对')
    parser.add_argument('--interval', default='4h', help='K线周期')
    parser.add_argument('--start', default='2020-01-01', help='开始日期')
    parser.add_argument('--end', default=None, help='结束日期（默认到现在）')
    args = parser.parse_args()

    df = fetch_klines(args.symbol, args.interval, args.start, args.end)

    # 保存
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    filename = f"{args.symbol}_{args.interval}.csv"
    out_path = DATA_DIR / filename
    df.to_csv(out_path)

    print(f"\n数据已保存: {out_path}")
    print(f"时间范围: {df.index[0]} ~ {df.index[-1]}")
    print(f"总行数: {len(df)}")
    print(f"\n前5行:")
    print(df.head())
    print(f"\n后5行:")
    print(df.tail())


if __name__ == '__main__':
    main()
