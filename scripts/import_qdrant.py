"""
将向量和元数据导入 Qdrant 向量数据库。
用法: python scripts/import_qdrant.py [--symbol BTCUSDT] [--interval 1h]
"""
import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import argparse
import json
import numpy as np
from qdrant_client import QdrantClient
from qdrant_client.models import Distance, VectorParams, PointStruct
from config.settings import (
    DATA_DIR, QDRANT_HOST, QDRANT_PORT, COLLECTION_NAME, EMBEDDING_DIM
)


def import_to_qdrant(symbol: str, interval: str):
    """导入向量到Qdrant"""
    # 加载数据
    meta_path = DATA_DIR / f"{symbol}_{interval}_patterns.json"
    embed_path = DATA_DIR / f"{symbol}_{interval}_embeddings.npy"

    if not meta_path.exists() or not embed_path.exists():
        print("数据文件不存在，请先运行 embed_charts.py")
        return

    with open(meta_path, 'r', encoding='utf-8') as f:
        patterns = json.load(f)

    vectors = np.load(embed_path)
    # 只导入已Embedding的部分（断点续传时向量数可能少于形态数）
    actual_count = min(len(patterns), len(vectors))
    patterns = patterns[:actual_count]
    print(f"加载 {actual_count} 个形态(总{len(vectors)}向量), 向量形状: {vectors.shape}")

    # 连接 Qdrant
    client = QdrantClient(host=QDRANT_HOST, port=QDRANT_PORT)

    # 创建/重建 collection
    collections = [c.name for c in client.get_collections().collections]
    if COLLECTION_NAME in collections:
        print(f"Collection '{COLLECTION_NAME}' 已存在，将重建...")
        client.delete_collection(COLLECTION_NAME)

    client.create_collection(
        collection_name=COLLECTION_NAME,
        vectors_config=VectorParams(
            size=EMBEDDING_DIM,
            distance=Distance.COSINE,
        ),
    )
    print(f"创建 collection: {COLLECTION_NAME} (dim={EMBEDDING_DIM}, cosine)")

    # 批量导入
    batch_size = 100
    total = len(patterns)
    for start in range(0, total, batch_size):
        end = min(start + batch_size, total)
        points = []
        for i in range(start, end):
            vec = vectors[i].tolist()
            # 跳过全零向量（embedding失败的）
            if all(v == 0.0 for v in vec[:10]):
                continue

            meta = patterns[i]
            points.append(PointStruct(
                id=i,
                vector=vec,
                payload={
                    'timestamp': meta['timestamp'],
                    'datetime': meta['datetime'],
                    'symbol': meta['symbol'],
                    'interval': meta['interval'],
                    'entry_price': meta['entry_price'],
                    'return_5': meta.get('return_5'),
                    'return_10': meta.get('return_10'),
                    'return_20': meta.get('return_20'),
                    'file': meta['file'],
                },
            ))

        if points:
            client.upsert(collection_name=COLLECTION_NAME, points=points)

        if (end) % 1000 == 0 or end == total:
            print(f"  导入进度: {end}/{total}")

    # 验证
    info = client.get_collection(COLLECTION_NAME)
    print(f"\n导入完成!")
    print(f"Collection: {COLLECTION_NAME}")
    print(f"向量数量: {info.points_count}")
    print(f"向量维度: {EMBEDDING_DIM}")


def main():
    parser = argparse.ArgumentParser(description='Import embeddings to Qdrant')
    parser.add_argument('--symbol', default='BTCUSDT')
    parser.add_argument('--interval', default='4h')
    args = parser.parse_args()

    import_to_qdrant(args.symbol, args.interval)


if __name__ == '__main__':
    main()
