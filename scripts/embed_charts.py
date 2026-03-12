"""
用 Gemini Embedding 2 对K线图片做批量向量化。
用法: python scripts/embed_charts.py [--symbol BTCUSDT] [--interval 1h]
"""
import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import argparse
import json
import time
import numpy as np
from pathlib import Path
from google import genai
from google.genai import types
from PIL import Image
from config.settings import (
    DATA_DIR, CHARTS_DIR, GEMINI_API_KEY, EMBEDDING_DIM
)


def embed_charts(symbol: str, interval: str):
    """批量Embedding K线图片"""
    if not GEMINI_API_KEY:
        print("错误: 请设置环境变量 GEMINI_API_KEY")
        print("  Windows: set GEMINI_API_KEY=your_key")
        print("  Linux:   export GEMINI_API_KEY=your_key")
        return

    client = genai.Client(api_key=GEMINI_API_KEY)

    # 加载元数据
    meta_path = DATA_DIR / f"{symbol}_{interval}_patterns.json"
    if not meta_path.exists():
        print(f"元数据不存在: {meta_path}")
        print("请先运行: python scripts/generate_charts.py")
        return

    with open(meta_path, 'r', encoding='utf-8') as f:
        patterns = json.load(f)

    print(f"加载 {len(patterns)} 个形态，开始Embedding...")

    chart_dir = CHARTS_DIR / symbol
    embeddings_path = DATA_DIR / f"{symbol}_{interval}_embeddings.npy"
    progress_path = DATA_DIR / f"{symbol}_{interval}_embed_progress.json"

    # 断点续传：检查已完成的进度
    start_idx = 0
    all_vectors = []
    if progress_path.exists():
        with open(progress_path, 'r') as f:
            progress = json.load(f)
        start_idx = progress.get('completed', 0)
        if embeddings_path.exists() and start_idx > 0:
            all_vectors = np.load(embeddings_path).tolist()[:start_idx]
            print(f"从断点续传: 已完成 {start_idx}/{len(patterns)}")

    failed = []
    for i in range(start_idx, len(patterns)):
        pat = patterns[i]
        img_path = chart_dir / pat['file']

        if not img_path.exists():
            print(f"  图片不存在，跳过: {img_path}")
            all_vectors.append([0.0] * EMBEDDING_DIM)
            failed.append(i)
            continue

        try:
            img = Image.open(img_path)
            result = client.models.embed_content(
                model='gemini-embedding-2-preview',
                contents=img,
                config=types.EmbedContentConfig(
                    output_dimensionality=EMBEDDING_DIM,
                ),
            )
            vector = result.embeddings[0].values
            all_vectors.append(vector)

        except Exception as e:
            error_msg = str(e)
            if 'quota' in error_msg.lower() or '429' in error_msg:
                print(f"  API限速，等待60秒... (已完成 {i}/{len(patterns)})")
                # 保存进度
                np.save(embeddings_path, np.array(all_vectors))
                with open(progress_path, 'w') as f:
                    json.dump({'completed': i}, f)
                time.sleep(60)
                # 重试当前
                try:
                    img = Image.open(img_path)
                    result = client.models.embed_content(
                        model='gemini-embedding-2-preview',
                        contents=img,
                        config=types.EmbedContentConfig(
                            output_dimensionality=EMBEDDING_DIM,
                        ),
                    )
                    all_vectors.append(result.embeddings[0].values)
                except Exception as e2:
                    print(f"  重试失败: {e2}")
                    all_vectors.append([0.0] * EMBEDDING_DIM)
                    failed.append(i)
            else:
                print(f"  Embedding失败 [{i}]: {e}")
                all_vectors.append([0.0] * EMBEDDING_DIM)
                failed.append(i)

        # 限速控制：免费层60RPM，每秒1个请求保守
        time.sleep(1.1)

        if (i + 1) % 100 == 0:
            print(f"  进度: {i + 1}/{len(patterns)}")
            # 定期保存
            np.save(embeddings_path, np.array(all_vectors))
            with open(progress_path, 'w') as f:
                json.dump({'completed': i + 1}, f)

    # 最终保存
    vectors_array = np.array(all_vectors)
    np.save(embeddings_path, vectors_array)

    # 清理进度文件
    if progress_path.exists():
        progress_path.unlink()

    print(f"\nEmbedding完成!")
    print(f"向量形状: {vectors_array.shape}")
    print(f"保存到: {embeddings_path}")
    if failed:
        print(f"失败数量: {len(failed)}")


def main():
    parser = argparse.ArgumentParser(description='Embed kline charts with Gemini')
    parser.add_argument('--symbol', default='BTCUSDT')
    parser.add_argument('--interval', default='4h')
    args = parser.parse_args()

    embed_charts(args.symbol, args.interval)


if __name__ == '__main__':
    main()
