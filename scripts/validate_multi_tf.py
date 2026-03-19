"""
多周期共振验证：同一时间点的1h/4h/1d三个周期形态检索，
只有方向一致才出信号，统计准确率。

用法: python scripts/validate_multi_tf.py [--samples 200] [--top_k 10]
"""
import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import argparse
import json
import random
import numpy as np
from qdrant_client import QdrantClient
from config.settings import (
    DATA_DIR, QDRANT_HOST, QDRANT_PORT, COLLECTION_NAME, TOP_K
)

# 多周期配置
TIMEFRAMES = ['1h', '4h', '1d']
COLLECTION_MAP = {
    '1h': f'{COLLECTION_NAME}_1h',
    '4h': COLLECTION_NAME,
    '1d': f'{COLLECTION_NAME}_1d',
}


def load_tf_data(symbol, interval):
    """加载某个周期的元数据和向量"""
    meta_path = DATA_DIR / f"{symbol}_{interval}_patterns.json"
    embed_path = DATA_DIR / f"{symbol}_{interval}_embeddings.npy"

    if not meta_path.exists() or not embed_path.exists():
        return None, None

    with open(meta_path, 'r', encoding='utf-8') as f:
        patterns = json.load(f)
    vectors = np.load(embed_path)
    return patterns, vectors


def find_closest_pattern(patterns, target_ts, max_gap=7200):
    """找到最接近目标时间戳的形态索引"""
    best_idx = None
    best_gap = float('inf')
    for i, p in enumerate(patterns):
        gap = abs(p['timestamp'] - target_ts)
        if gap < best_gap:
            best_gap = gap
            best_idx = i
    if best_gap <= max_gap:
        return best_idx
    return None


def query_direction(client, collection_name, query_vector, query_ts, top_k):
    """用向量检索并返回方向投票"""
    try:
        results = client.query_points(
            collection_name=collection_name,
            query=query_vector,
            limit=top_k + 20,
        )
        filtered = [
            r for r in results.points
            if abs(r.payload['timestamp'] - query_ts) > 48 * 3600
        ][:top_k]

        if len(filtered) < 3:
            return None, 0

        up_count = sum(1 for r in filtered if (r.payload.get('return_20') or 0) > 0)
        hit_rate = up_count / len(filtered)
        return hit_rate, len(filtered)

    except Exception:
        return None, 0


def validate_multi_tf(symbol: str, num_samples: int, top_k: int):
    """多周期共振验证"""
    # 加载所有周期数据
    tf_data = {}
    for tf in TIMEFRAMES:
        patterns, vectors = load_tf_data(symbol, tf)
        if patterns is None:
            print(f"跳过 {tf}：数据不存在")
            continue
        tf_data[tf] = {'patterns': patterns, 'vectors': vectors}
        print(f"  {tf}: {len(patterns)} 个形态, {vectors.shape} 向量")

    available_tfs = list(tf_data.keys())
    print(f"\n可用周期: {available_tfs}")

    if '4h' not in tf_data:
        print("4h 数据必须存在")
        return

    client = QdrantClient(host=QDRANT_HOST, port=QDRANT_PORT)

    # 用 4h 形态作为基准，随机抽样
    patterns_4h = tf_data['4h']['patterns']
    vectors_4h = tf_data['4h']['vectors']

    valid_indices = [
        i for i, p in enumerate(patterns_4h)
        if p.get('return_20') is not None
        and i < len(vectors_4h)
        and not all(v == 0.0 for v in vectors_4h[i][:10])
    ]

    if num_samples < len(valid_indices):
        sample_indices = sorted(random.sample(valid_indices, num_samples))
    else:
        sample_indices = valid_indices
        num_samples = len(sample_indices)

    print(f"\n验证样本: {num_samples}")

    # 统计
    stats = {
        'single_4h': {'correct': 0, 'total': 0},
        'consensus_2': {'correct': 0, 'total': 0, 'long_c': 0, 'long_t': 0, 'short_c': 0, 'short_t': 0},
        'consensus_all': {'correct': 0, 'total': 0, 'long_c': 0, 'long_t': 0, 'short_c': 0, 'short_t': 0},
    }

    for idx in sample_indices:
        pat_4h = patterns_4h[idx]
        query_ts = pat_4h['timestamp']
        actual_return = pat_4h['return_20']
        actual_up = actual_return > 0

        # 各周期检索方向
        tf_results = {}
        for tf in available_tfs:
            coll_name = COLLECTION_MAP[tf]

            if tf == '4h':
                qvec = vectors_4h[idx].tolist()
            else:
                # 找对应时间点最近的形态
                tf_patterns = tf_data[tf]['patterns']
                tf_vectors = tf_data[tf]['vectors']
                closest_idx = find_closest_pattern(tf_patterns, query_ts,
                                                    max_gap=3600 if tf == '1h' else 86400)
                if closest_idx is None or closest_idx >= len(tf_vectors):
                    continue
                if all(v == 0.0 for v in tf_vectors[closest_idx][:10]):
                    continue
                qvec = tf_vectors[closest_idx].tolist()

            hit_rate, count = query_direction(client, coll_name, qvec, query_ts, top_k)
            if hit_rate is not None:
                tf_results[tf] = hit_rate

        # 单4h统计
        if '4h' in tf_results:
            predicted_up_4h = tf_results['4h'] > 0.5
            stats['single_4h']['total'] += 1
            if predicted_up_4h == actual_up:
                stats['single_4h']['correct'] += 1

        # 多周期共识（≥2个周期方向一致）
        if len(tf_results) >= 2:
            up_votes = sum(1 for hr in tf_results.values() if hr > 0.5)
            down_votes = len(tf_results) - up_votes

            if up_votes >= 2 or down_votes >= 2:
                consensus_up = up_votes > down_votes
                stats['consensus_2']['total'] += 1
                if consensus_up == actual_up:
                    stats['consensus_2']['correct'] += 1

                # LONG/SHORT 强信号
                max_agreement = max(up_votes, down_votes) / len(tf_results)
                if max_agreement >= 0.9:  # 几乎全部一致
                    if consensus_up:
                        stats['consensus_2']['long_t'] += 1
                        if actual_up:
                            stats['consensus_2']['long_c'] += 1
                    else:
                        stats['consensus_2']['short_t'] += 1
                        if not actual_up:
                            stats['consensus_2']['short_c'] += 1

        # 全部周期一致
        if len(tf_results) == len(available_tfs) and len(available_tfs) >= 2:
            all_up = all(hr > 0.5 for hr in tf_results.values())
            all_down = all(hr <= 0.5 for hr in tf_results.values())

            if all_up or all_down:
                consensus_up = all_up
                stats['consensus_all']['total'] += 1
                if consensus_up == actual_up:
                    stats['consensus_all']['correct'] += 1

                # LONG/SHORT
                if all_up:
                    stats['consensus_all']['long_t'] += 1
                    if actual_up:
                        stats['consensus_all']['long_c'] += 1
                else:
                    stats['consensus_all']['short_t'] += 1
                    if not actual_up:
                        stats['consensus_all']['short_c'] += 1

    # 输出结果
    print(f"\n{'='*70}")
    print(f"多周期共振验证结果 (Top-{top_k}, 可用周期: {available_tfs})")
    print(f"{'='*70}")

    for label, s in stats.items():
        if s['total'] == 0:
            continue
        acc = s['correct'] / s['total'] * 100
        print(f"\n  [{label}]")
        print(f"    总信号: {s['total']}, 准确率: {acc:.1f}%")
        if 'long_t' in s and s['long_t'] > 0:
            la = s['long_c'] / s['long_t'] * 100
            print(f"    强LONG: {s['long_c']}/{s['long_t']} = {la:.1f}%")
        if 'short_t' in s and s['short_t'] > 0:
            sa = s['short_c'] / s['short_t'] * 100
            print(f"    强SHORT: {s['short_c']}/{s['short_t']} = {sa:.1f}%")

    # 信号频率
    if stats['consensus_all']['total'] > 0:
        freq = stats['consensus_all']['total'] / num_samples * 100
        print(f"\n  全周期共振信号占比: {freq:.1f}%")


def main():
    parser = argparse.ArgumentParser(description='Multi-timeframe consensus validation')
    parser.add_argument('--symbol', default='BTCUSDT')
    parser.add_argument('--samples', type=int, default=200)
    parser.add_argument('--top_k', type=int, default=10)
    args = parser.parse_args()

    random.seed(42)
    validate_multi_tf(args.symbol, args.samples, args.top_k)


if __name__ == '__main__':
    main()
