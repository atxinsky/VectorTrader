"""
批量验证：对比不同策略的准确率。
不调API，直接用Qdrant中的现有向量做离线验证。

用法: python scripts/validate.py [--samples 200] [--top_k 10]
"""
import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import argparse
import json
import random
import numpy as np
from qdrant_client import QdrantClient
from qdrant_client.models import Filter, FieldCondition, MatchValue
from config.settings import (
    DATA_DIR, QDRANT_HOST, QDRANT_PORT, COLLECTION_NAME, TOP_K
)


def validate(symbol: str, interval: str, num_samples: int, top_k: int):
    """批量验证准确率"""
    # 加载元数据
    meta_path = DATA_DIR / f"{symbol}_{interval}_patterns.json"
    embed_path = DATA_DIR / f"{symbol}_{interval}_embeddings.npy"

    with open(meta_path, 'r', encoding='utf-8') as f:
        patterns = json.load(f)

    vectors = np.load(embed_path)

    client = QdrantClient(host=QDRANT_HOST, port=QDRANT_PORT)

    # 筛选有效样本（有return_20且非零向量）
    valid_indices = [
        i for i, p in enumerate(patterns)
        if p.get('return_20') is not None
        and i < len(vectors)
        and not all(v == 0.0 for v in vectors[i][:10])
    ]

    # 随机抽样
    if num_samples < len(valid_indices):
        sample_indices = sorted(random.sample(valid_indices, num_samples))
    else:
        sample_indices = valid_indices
        num_samples = len(sample_indices)

    print(f"验证样本: {num_samples}/{len(valid_indices)} (Top-{top_k})")
    print(f"品种: {symbol}, 周期: {interval}")
    has_regime = 'regime' in patterns[0] if patterns else False
    print(f"环境标签: {'有' if has_regime else '无'}")
    print()

    # ========== 策略1：无过滤（基线） ==========
    results_baseline = _run_validation(
        client, vectors, patterns, sample_indices, top_k,
        filter_regime=False, label="基线(无过滤)"
    )

    # ========== 策略2：按环境过滤 ==========
    if has_regime:
        results_regime = _run_validation(
            client, vectors, patterns, sample_indices, top_k,
            filter_regime=True, label="环境过滤"
        )
    else:
        print("跳过环境过滤验证（未打标签，请先运行 label_regime.py）")
        results_regime = None

    # ========== 汇总对比 ==========
    print(f"\n{'='*70}")
    print(f"对比汇总 (样本数: {num_samples}, Top-{top_k})")
    print(f"{'='*70}")

    _print_summary("基线(无过滤)", results_baseline)
    if results_regime:
        _print_summary("环境过滤", results_regime)

        # 计算提升
        delta = results_regime['accuracy'] - results_baseline['accuracy']
        print(f"\n环境过滤 vs 基线: {delta:+.1f}% 准确率变化")


def _run_validation(client, vectors, patterns, sample_indices, top_k,
                    filter_regime: bool, label: str):
    """运行一轮验证"""
    correct = 0
    total = 0
    long_correct = 0
    long_total = 0
    short_correct = 0
    short_total = 0
    skipped = 0

    # 分环境统计
    regime_stats = {'uptrend': [0, 0], 'downtrend': [0, 0], 'ranging': [0, 0]}

    for idx in sample_indices:
        pat = patterns[idx]
        query_vector = vectors[idx].tolist()
        query_ts = pat['timestamp']
        actual_return = pat['return_20']
        actual_direction = 'up' if actual_return > 0 else 'down'
        query_regime = pat.get('regime', 'ranging')

        # 构建过滤条件
        qdrant_filter = None
        if filter_regime and query_regime:
            qdrant_filter = Filter(
                must=[FieldCondition(key="regime", match=MatchValue(value=query_regime))]
            )

        try:
            results = client.query_points(
                collection_name=COLLECTION_NAME,
                query=query_vector,
                query_filter=qdrant_filter,
                limit=top_k + 20,
            )

            # 过滤时间邻近样本
            filtered = [
                r for r in results.points
                if abs(r.payload['timestamp'] - query_ts) > 48 * 3600
            ][:top_k]

            if len(filtered) < 3:
                skipped += 1
                continue

            # 统计Top-K方向
            up_count = sum(
                1 for r in filtered
                if (r.payload.get('return_20') or 0) > 0
            )
            predicted_direction = 'up' if up_count > len(filtered) / 2 else 'down'

            total += 1
            if predicted_direction == actual_direction:
                correct += 1

            # LONG/SHORT 分别统计
            hit_rate = up_count / len(filtered)
            if hit_rate >= 0.7:
                long_total += 1
                if actual_return > 0:
                    long_correct += 1
            elif hit_rate <= 0.3:
                short_total += 1
                if actual_return < 0:
                    short_correct += 1

            # 分环境统计
            if query_regime in regime_stats:
                regime_stats[query_regime][1] += 1
                if predicted_direction == actual_direction:
                    regime_stats[query_regime][0] += 1

        except Exception as e:
            skipped += 1

    accuracy = correct / total * 100 if total > 0 else 0
    long_acc = long_correct / long_total * 100 if long_total > 0 else 0
    short_acc = short_correct / short_total * 100 if short_total > 0 else 0

    print(f"\n--- {label} ---")
    print(f"  有效样本: {total} (跳过: {skipped})")
    print(f"  整体准确率: {accuracy:.1f}%")
    print(f"  LONG信号: {long_correct}/{long_total} = {long_acc:.1f}%")
    print(f"  SHORT信号: {short_correct}/{short_total} = {short_acc:.1f}%")

    for regime, (c, t) in regime_stats.items():
        if t > 0:
            print(f"  [{regime}] {c}/{t} = {c/t*100:.1f}%")

    return {
        'accuracy': accuracy,
        'total': total,
        'correct': correct,
        'long_acc': long_acc,
        'long_total': long_total,
        'short_acc': short_acc,
        'short_total': short_total,
        'regime_stats': regime_stats,
    }


def _print_summary(label, result):
    acc = result['accuracy']
    lt = result['long_total']
    la = result['long_acc']
    st = result['short_total']
    sa = result['short_acc']
    print(f"  {label:15s} | 准确率 {acc:5.1f}% | LONG {la:5.1f}% ({lt:3d}次) | SHORT {sa:5.1f}% ({st:3d}次)")


def main():
    parser = argparse.ArgumentParser(description='Validate pattern search strategies')
    parser.add_argument('--symbol', default='BTCUSDT')
    parser.add_argument('--interval', default='4h')
    parser.add_argument('--samples', type=int, default=200)
    parser.add_argument('--top_k', type=int, default=10)
    args = parser.parse_args()

    random.seed(42)  # 可复现
    validate(args.symbol, args.interval, args.samples, args.top_k)


if __name__ == '__main__':
    main()
