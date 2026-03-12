"""
形态检索验证：取当前/指定时间的K线形态，检索最相似的历史形态，输出对比。
用法:
  python scripts/search_pattern.py                     # 检索最新形态
  python scripts/search_pattern.py --offset 100        # 检索倒数第100个窗口
  python scripts/search_pattern.py --top_k 10          # 返回Top-10
"""
import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import argparse
import json
import pandas as pd
import numpy as np
import mplfinance as mpf
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
from PIL import Image
from google import genai
from google.genai import types
from qdrant_client import QdrantClient
from config.settings import (
    DATA_DIR, CHARTS_DIR, GEMINI_API_KEY, EMBEDDING_DIM,
    QDRANT_HOST, QDRANT_PORT, COLLECTION_NAME,
    WINDOW_SIZE, TOP_K, SIMILARITY_THRESHOLD, CONFIDENCE_THRESHOLD
)


def search_similar(symbol: str, interval: str, offset: int, top_k: int):
    """检索相似形态"""
    # 加载K线数据
    csv_path = DATA_DIR / f"{symbol}_{interval}.csv"
    df = pd.read_csv(csv_path, index_col='datetime', parse_dates=True)

    # 截取查询窗口
    start_idx = len(df) - WINDOW_SIZE - offset
    query_chunk = df.iloc[start_idx:start_idx + WINDOW_SIZE].copy()
    query_time = query_chunk.index[-1]
    query_price = query_chunk['close'].iloc[-1]

    print(f"查询形态: {query_chunk.index[0]} ~ {query_time}")
    print(f"当前价格: {query_price:.2f}")

    # 生成查询图片
    mc = mpf.make_marketcolors(
        up='#26a69a', down='#ef5350',
        edge={'up': '#26a69a', 'down': '#ef5350'},
        wick={'up': '#26a69a', 'down': '#ef5350'},
        volume={'up': '#26a69a80', 'down': '#ef535080'},
    )
    style = mpf.make_mpf_style(
        marketcolors=mc, figcolor='white', facecolor='white',
        gridstyle='', y_on_right=False,
    )

    query_img_path = CHARTS_DIR / "query_current.png"
    mpf.plot(
        query_chunk, type='candle', volume=True, style=style,
        axisoff=True, figsize=(4, 3),
        savefig=dict(fname=str(query_img_path), dpi=72,
                     bbox_inches='tight', pad_inches=0.05),
        tight_layout=True,
    )

    # Embedding
    gemini_client = genai.Client(api_key=GEMINI_API_KEY)
    img = Image.open(query_img_path)
    result = gemini_client.models.embed_content(
        model='gemini-embedding-2-preview',
        contents=img,
        config=types.EmbedContentConfig(
            output_dimensionality=EMBEDDING_DIM,
        ),
    )
    query_vector = result.embeddings[0].values

    # Qdrant 检索
    client = QdrantClient(host=QDRANT_HOST, port=QDRANT_PORT)
    results = client.query_points(
        collection_name=COLLECTION_NAME,
        query=query_vector,
        limit=top_k + 10,  # 多取一些，过滤掉时间太近的
    )

    # 过滤：排除与查询时间太近的样本（±48小时）
    query_ts = int(query_time.timestamp())
    filtered = [
        r for r in results.points
        if abs(r.payload['timestamp'] - query_ts) > 48 * 3600
    ][:top_k]

    # 输出结果
    print(f"\n{'='*60}")
    print(f"Top-{len(filtered)} 相似形态")
    print(f"{'='*60}")

    up_count = 0
    down_count = 0
    returns_5 = []
    returns_10 = []
    returns_20 = []

    for rank, hit in enumerate(filtered, 1):
        p = hit.payload
        similarity = hit.score
        ret5 = p.get('return_5')
        ret10 = p.get('return_10')
        ret20 = p.get('return_20')

        direction = "涨" if (ret20 or ret10 or ret5 or 0) > 0 else "跌"
        if direction == "涨":
            up_count += 1
        else:
            down_count += 1

        if ret5 is not None:
            returns_5.append(ret5)
        if ret10 is not None:
            returns_10.append(ret10)
        if ret20 is not None:
            returns_20.append(ret20)

        print(f"\n  #{rank} | 相似度: {similarity:.4f} | {p['datetime']}")
        print(f"       价格: {p['entry_price']:.2f}")
        print(f"       后续5h: {ret5:+.2f}%" if ret5 else "       后续5h: N/A")
        print(f"       后续10h: {ret10:+.2f}%" if ret10 else "       后续10h: N/A")
        print(f"       后续20h: {ret20:+.2f}%" if ret20 else "       后续20h: N/A")
        print(f"       图片: {p['file']}")

    # 统计汇总
    total = up_count + down_count
    hit_rate = up_count / total * 100 if total > 0 else 0

    print(f"\n{'='*60}")
    print(f"统计汇总")
    print(f"{'='*60}")
    print(f"  样本数: {total}")
    print(f"  看涨: {up_count} ({hit_rate:.1f}%)")
    print(f"  看跌: {down_count} ({100 - hit_rate:.1f}%)")
    if returns_5:
        print(f"  平均后续5h收益: {np.mean(returns_5):+.3f}%")
    if returns_10:
        print(f"  平均后续10h收益: {np.mean(returns_10):+.3f}%")
    if returns_20:
        print(f"  平均后续20h收益: {np.mean(returns_20):+.3f}%")

    avg_sim = np.mean([r.score for r in filtered])
    print(f"  平均相似度: {avg_sim:.4f}")

    # 信号判断
    print(f"\n{'='*60}")
    if avg_sim >= SIMILARITY_THRESHOLD:
        if hit_rate >= CONFIDENCE_THRESHOLD * 100:
            print(f"  信号: LONG (胜率{hit_rate:.0f}%, 相似度{avg_sim:.2f})")
        elif hit_rate <= (1 - CONFIDENCE_THRESHOLD) * 100:
            print(f"  信号: SHORT (跌率{100 - hit_rate:.0f}%, 相似度{avg_sim:.2f})")
        else:
            print(f"  信号: 无 (胜率{hit_rate:.0f}%在中间区域，方向不明确)")
    else:
        print(f"  信号: 无 (相似度{avg_sim:.2f} < 阈值{SIMILARITY_THRESHOLD})")
    print(f"{'='*60}")

    # 生成对比图
    _generate_comparison(query_img_path, filtered, symbol)


def _generate_comparison(query_path, results, symbol):
    """生成查询形态 vs Top-5 相似形态的对比图"""
    top5 = results[:5]
    chart_dir = CHARTS_DIR / symbol

    fig, axes = plt.subplots(1, min(6, len(top5) + 1), figsize=(20, 4))
    if len(top5) + 1 == 1:
        axes = [axes]

    # 查询图
    query_img = mpimg.imread(str(query_path))
    axes[0].imshow(query_img)
    axes[0].set_title('QUERY (current)', fontsize=9, fontweight='bold', color='#2e6ab0')
    axes[0].axis('off')

    # 相似形态图
    for i, hit in enumerate(top5):
        if i + 1 >= len(axes):
            break
        img_file = chart_dir / hit.payload['file']
        if img_file.exists():
            img = mpimg.imread(str(img_file))
            axes[i + 1].imshow(img)
        ret20 = hit.payload.get('return_20', 0) or 0
        color = '#26a69a' if ret20 > 0 else '#ef5350'
        axes[i + 1].set_title(
            f"#{i+1} sim={hit.score:.3f}\n{hit.payload['datetime'][:10]} → {ret20:+.2f}%",
            fontsize=7, color=color
        )
        axes[i + 1].axis('off')

    plt.suptitle(f'VectorTrader - Pattern Search ({symbol})', fontsize=11, fontweight='bold')
    plt.tight_layout()

    out_path = CHARTS_DIR / "comparison_latest.png"
    plt.savefig(str(out_path), dpi=120, bbox_inches='tight')
    plt.close()
    print(f"\n对比图已保存: {out_path}")


def main():
    parser = argparse.ArgumentParser(description='Search similar kline patterns')
    parser.add_argument('--symbol', default='BTCUSDT')
    parser.add_argument('--interval', default='4h')
    parser.add_argument('--offset', type=int, default=0, help='从最新往前偏移N个窗口')
    parser.add_argument('--top_k', type=int, default=TOP_K)
    args = parser.parse_args()

    search_similar(args.symbol, args.interval, args.offset, args.top_k)


if __name__ == '__main__':
    main()
