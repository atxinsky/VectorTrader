# VectorTrader — K线形态 Embedding 交易系统

基于 Gemini Embedding 2 的多模态形态识别 + Qdrant 向量检索交易系统。

## 原理

> "刻舟求剑" — 当前K线形态和历史上哪些形态最像？那些形态后来涨了还是跌了？

1. 用 mplfinance 把每段K线生成一张**纯净图片**（去坐标轴，消除绝对价格影响）
2. 用 Gemini Embedding 2 把图片转成 **768维向量**
3. 存入 Qdrant 向量数据库
4. 新形态进来 → Embedding → 检索最相似的历史形态 → 统计后续涨跌 → 输出信号

## 技术栈

| 组件 | 选型 |
|------|------|
| Embedding 模型 | Gemini Embedding 2 (`gemini-embedding-2-preview`) |
| K线图生成 | mplfinance (`axisoff=True` 纯净图片) |
| 向量数据库 | Qdrant (Docker 本地) |
| 历史数据 | Binance API (`python-binance`) |
| 语言 | Python |

## 项目结构

```
VectorTrader/
├── config/
│   ├── __init__.py
│   └── settings.py          # 全局配置（从 .env 读取 API Key）
├── scripts/
│   ├── fetch_klines.py      # Step 1: 拉取 Binance 历史K线
│   ├── generate_charts.py   # Step 2: 滑动窗口生成形态图片
│   ├── embed_charts.py      # Step 3: Gemini Embedding 批量向量化
│   ├── import_qdrant.py     # Step 4: 导入 Qdrant 向量库
│   └── search_pattern.py    # Step 5: 相似形态检索验证
├── run_pipeline.py           # 一键运行全流程
├── data/                     # K线CSV + 向量npy（gitignore）
├── charts/                   # 形态图片（gitignore）
├── .env                      # API Key（gitignore）
└── requirements.txt
```

## 快速开始

```bash
# 1. 安装依赖
pip install -r requirements.txt

# 2. 配置 API Key
echo "GEMINI_API_KEY=your_key" > .env

# 3. 启动 Qdrant
docker run -d --name qdrant -p 6333:6333 qdrant/qdrant

# 4. 一键运行全流程
python run_pipeline.py

# 或分步运行
python scripts/fetch_klines.py          # 拉数据
python scripts/generate_charts.py       # 生成图片
python scripts/embed_charts.py          # Embedding（支持断点续传）
python scripts/import_qdrant.py         # 导入Qdrant
python scripts/search_pattern.py        # 检索验证
```

## 当前参数

| 参数 | 值 | 说明 |
|------|-----|------|
| 周期 | 4h | 每根K线 = 4小时 |
| 窗口 | 20根 | 每个形态 ≈ 3.3天 |
| 步长 | 3根 | 每12小时产生一个新样本 |
| 向量维度 | 768 | 从3072截断 |
| 品种 | BTCUSDT | Phase 1 只做BTC |
| 数据范围 | 2020-01 ~ 2026-03 | 约13575根K线，4512个形态 |

## 2026-03-12 进度记录

### 已完成

1. **数据拉取** — BTC/USDT 4h K线，2020-01至今，13575根
2. **图片生成** — 滑动窗口生成 4512 张形态图片（400x300 PNG，含量能柱）
3. **Embedding** — 完成 813/4512 张（触发API日限额后停止）
4. **Qdrant导入** — 887个有效向量已导入（过滤26个零向量）
5. **初步检索验证** — 用库内已有向量做了离线检索测试

### 初步验证结果

查询形态：2021-03-19 BTC $58,831（实际后续20h跌5.62%）

| 排名 | 相似度 | 日期 | 后续20h |
|------|--------|------|---------|
| #1 | 0.9938 | 2020-12-15 | +17.98% |
| #2 | 0.9937 | 2021-02-08 | +21.13% |
| #3 | 0.9936 | 2020-09-10 | +0.60% |
| #4 | 0.9934 | 2020-01-18 | -2.16% |
| #5 | 0.9933 | 2020-05-09 | -8.78% |

统计：6涨4跌(60%)，平均收益+3.49%，平均相似度0.9934

**观察：**
- 相似度极高（0.993+），Embedding 确实捕捉到了形态特征
- 但区分度太低（Top-1到Top-10仅差0.0007），887样本不够稀疏
- 这次查询实际跌了但系统给出看涨 — 样本量不足时结论不可靠

### API 限额问题

Gemini Embedding 2 免费层限制：

| 限制 | 数值 |
|------|------|
| RPM（每分钟） | 60次 |
| **RPD（每天）** | **1000次/模型** |
| 配额重置 | 太平洋时间午夜（北京时间下午3-4点） |

当前 4512 张图片，按 1000/天 的速度需要约 **5天** 跑完。
脚本支持断点续传，每天运行 `python scripts/embed_charts.py` 即可继续。

### 明天继续

```bash
# 继续 Embedding（自动从断点恢复）
cd D:\VectorTrader
python scripts/embed_charts.py

# Embedding 全部完成后
python scripts/import_qdrant.py
python scripts/search_pattern.py --top_k 20
```

## 后续计划

- **Phase 1 剩余**：跑完全部 Embedding → 全量验证 → 回测统计框架
- **Phase 2**：多品种（ETH、螺纹钢、原油）+ 实时信号推送
- **Phase 3**：OKX 模拟盘自动交易
- **Phase 4**：融入 Banbot 实盘
