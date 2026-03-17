"""
从CSV K线数据生成K线图片。
滑动窗口截取，每张图片20根K线+量能柱，保留坐标轴为Embedding提供语义信息。
用法: python scripts/generate_charts.py [--symbol BTCUSDT] [--interval 4h]
"""
import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import argparse
import json
import pandas as pd
import mplfinance as mpf
from pathlib import Path
from config.settings import DATA_DIR, CHARTS_DIR, WINDOW_SIZE, STEP_SIZE


def generate_charts(symbol: str, interval: str, window: int, step: int):
    """滑动窗口生成K线图片，返回形态元数据列表"""
    csv_path = DATA_DIR / f"{symbol}_{interval}.csv"
    if not csv_path.exists():
        print(f"数据文件不存在: {csv_path}")
        print("请先运行: python scripts/fetch_klines.py")
        return []

    df = pd.read_csv(csv_path, index_col='datetime', parse_dates=True)
    print(f"加载 {len(df)} 根K线: {symbol} {interval}")

    # 图片输出目录
    out_dir = CHARTS_DIR / symbol
    out_dir.mkdir(parents=True, exist_ok=True)

    # mplfinance 样式：黑底白线，纯净无干扰
    mc = mpf.make_marketcolors(
        up='#26a69a', down='#ef5350',
        edge={'up': '#26a69a', 'down': '#ef5350'},
        wick={'up': '#26a69a', 'down': '#ef5350'},
        volume={'up': '#26a69a80', 'down': '#ef535080'},
    )
    style = mpf.make_mpf_style(
        marketcolors=mc,
        figcolor='white',
        facecolor='white',
        gridstyle='',
        y_on_right=False,
    )

    patterns = []
    total_windows = (len(df) - window) // step + 1
    print(f"将生成 {total_windows} 个形态图片...")

    for i in range(0, len(df) - window - window, step):
        # 当前窗口
        chunk = df.iloc[i:i + window].copy()
        # 后续窗口（用于计算后续收益）
        future = df.iloc[i + window:i + window + window]

        if len(future) < 5:
            break

        # 计算后续收益（后续20根K线的涨跌幅）
        entry_price = chunk['close'].iloc[-1]
        future_returns = []
        for j in [5, 10, 20]:
            if j <= len(future):
                ret = (future['close'].iloc[j - 1] - entry_price) / entry_price * 100
                future_returns.append(round(ret, 3))
            else:
                future_returns.append(None)

        # 文件名：用时间戳
        ts = int(chunk.index[-1].timestamp())
        filename = f"pat_{ts}.png"
        filepath = out_dir / filename

        # 计算窗口内趋势
        trend_pct = (chunk['close'].iloc[-1] - chunk['close'].iloc[0]) / chunk['close'].iloc[0] * 100

        # 生成图片（保留坐标轴，为多模态Embedding提供语义信息）
        save_config = dict(
            fname=str(filepath),
            dpi=100,
            bbox_inches='tight',
            pad_inches=0.1,
        )

        mpf.plot(
            chunk,
            type='candle',
            volume=True,
            style=style,
            title=f'Pattern pat_{ts} - Trend: {trend_pct:.2f}%',
            figsize=(5, 4),
            savefig=save_config,
            tight_layout=True,
        )

        # 记录元数据
        pattern_meta = {
            'file': filename,
            'timestamp': ts,
            'datetime': str(chunk.index[-1]),
            'symbol': symbol,
            'interval': interval,
            'entry_price': round(entry_price, 2),
            'return_5': future_returns[0],
            'return_10': future_returns[1],
            'return_20': future_returns[2],
        }
        patterns.append(pattern_meta)

        if len(patterns) % 500 == 0:
            print(f"  已生成 {len(patterns)}/{total_windows} 张图片...")

    # 保存元数据
    meta_path = DATA_DIR / f"{symbol}_{interval}_patterns.json"
    with open(meta_path, 'w', encoding='utf-8') as f:
        json.dump(patterns, f, ensure_ascii=False, indent=2)

    print(f"\n生成完成: {len(patterns)} 张图片")
    print(f"图片目录: {out_dir}")
    print(f"元数据: {meta_path}")
    return patterns


def main():
    parser = argparse.ArgumentParser(description='Generate kline chart images')
    parser.add_argument('--symbol', default='BTCUSDT')
    parser.add_argument('--interval', default='4h')
    parser.add_argument('--window', type=int, default=WINDOW_SIZE)
    parser.add_argument('--step', type=int, default=STEP_SIZE)
    args = parser.parse_args()

    import matplotlib
    matplotlib.use('Agg')  # 无GUI后端

    generate_charts(args.symbol, args.interval, args.window, args.step)


if __name__ == '__main__':
    main()
