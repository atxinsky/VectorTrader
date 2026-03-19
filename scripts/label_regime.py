"""
给每个形态打市场环境标签：uptrend / downtrend / ranging
基于形态窗口起始位置前 LOOKBACK 根K线的价格变化率判断。
用法: python scripts/label_regime.py [--symbol BTCUSDT] [--interval 4h]
"""
import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import argparse
import json
import pandas as pd
import numpy as np
from config.settings import DATA_DIR, WINDOW_SIZE, STEP_SIZE

# 环境判断参数
LOOKBACK = 60       # 往前看60根K线判断趋势（4h×60=10天）
TREND_THRESHOLD = 2.0  # SMA斜率超过2%算趋势，否则震荡


def classify_regime(df: pd.DataFrame, pattern_start_idx: int, lookback: int, threshold: float) -> str:
    """根据形态窗口前的K线判断市场环境 — SMA斜率法"""
    regime_start = max(0, pattern_start_idx - lookback)
    regime_end = pattern_start_idx

    if regime_end - regime_start < 20:
        return 'ranging'

    regime_slice = df.iloc[regime_start:regime_end]

    # 用SMA20的斜率判断趋势（比简单首尾涨跌更稳定）
    closes = regime_slice['close'].values
    sma20 = np.convolve(closes, np.ones(20)/20, mode='valid')

    if len(sma20) < 5:
        return 'ranging'

    # SMA斜率：最近5个SMA值的线性回归斜率，归一化为百分比
    recent_sma = sma20[-5:]
    slope = (recent_sma[-1] - recent_sma[0]) / recent_sma[0] * 100

    # 同时看ATR占比判断波动率
    highs = regime_slice['high'].values[-20:]
    lows = regime_slice['low'].values[-20:]
    atr = np.mean(highs - lows)
    atr_pct = atr / closes[-1] * 100

    if slope > threshold:
        return 'uptrend'
    elif slope < -threshold:
        return 'downtrend'
    else:
        return 'ranging'


def label_regime(symbol: str, interval: str):
    """给 patterns.json 中每个形态添加 regime 标签"""
    csv_path = DATA_DIR / f"{symbol}_{interval}.csv"
    meta_path = DATA_DIR / f"{symbol}_{interval}_patterns.json"

    if not csv_path.exists() or not meta_path.exists():
        print(f"数据不存在: {csv_path} 或 {meta_path}")
        return

    df = pd.read_csv(csv_path, index_col='datetime', parse_dates=True)
    with open(meta_path, 'r', encoding='utf-8') as f:
        patterns = json.load(f)

    print(f"加载 {len(df)} 根K线, {len(patterns)} 个形态")
    print(f"参数: lookback={LOOKBACK}, threshold={TREND_THRESHOLD}%")

    # 构建时间戳→索引映射
    ts_to_idx = {}
    for idx, row_time in enumerate(df.index):
        ts_to_idx[int(row_time.timestamp())] = idx

    counts = {'uptrend': 0, 'downtrend': 0, 'ranging': 0}

    for pat in patterns:
        ts = pat['timestamp']
        # 找到形态最后一根K线在df中的位置
        if ts in ts_to_idx:
            last_idx = ts_to_idx[ts]
            pattern_start_idx = last_idx - WINDOW_SIZE + 1
        else:
            pat['regime'] = 'ranging'
            counts['ranging'] += 1
            continue

        regime = classify_regime(df, pattern_start_idx, LOOKBACK, TREND_THRESHOLD)
        pat['regime'] = regime
        counts[regime] += 1

    # 保存
    with open(meta_path, 'w', encoding='utf-8') as f:
        json.dump(patterns, f, ensure_ascii=False, indent=2)

    print(f"\n标签分布:")
    for regime, count in counts.items():
        pct = count / len(patterns) * 100
        print(f"  {regime}: {count} ({pct:.1f}%)")
    print(f"\n已保存到: {meta_path}")


def main():
    parser = argparse.ArgumentParser(description='Label market regime for patterns')
    parser.add_argument('--symbol', default='BTCUSDT')
    parser.add_argument('--interval', default='4h')
    args = parser.parse_args()

    label_regime(args.symbol, args.interval)


if __name__ == '__main__':
    main()
