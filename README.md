# VectorTrader — K线形态 Embedding 交易系统

基于 Gemini Embedding 2 的多模态形态识别 + Qdrant 向量检索交易系统。

## 原理

> "刻舟求剑" — 当前K线形态和历史上哪些形态最像？那些形态后来涨了还是跌了？

1. 用 mplfinance 生成K线图片（**保留坐标轴**，为多模态模型提供语义信息）
2. 用 Gemini Embedding 2 把图片转成 **3072维向量**
3. 存入 Qdrant 向量数据库
4. 新形态进来 → Embedding → 检索最相似的历史形态 → 统计后续涨跌 → 输出信号

## 技术栈

| 组件 | 选型 |
|------|------|
| Embedding 模型 | Gemini Embedding 2 (`gemini-embedding-2-preview`) |
| K线图生成 | mplfinance（保留 Price/Volume/时间轴 + Trend% 标题）|
| 向量数据库 | Qdrant (Docker 本地) |
| 历史数据 | Binance API (`python-binance`) |
| 语言 | Python |

## 当前参数

| 参数 | 值 | 说明 |
|------|-----|------|
| 周期 | 4h | 每根K线 = 4小时 |
| 窗口 | 20根 | 每个形态 ≈ 3.3天 |
| 步长 | 3根 | 每12小时产生一个新样本 |
| 向量维度 | 3072 | 完整维度，不截断 |
| 品种 | BTCUSDT | Phase 1 只做BTC |
| 数据范围 | 2020-01 ~ 2026-03 | 13575根K线，4512个形态，4480个有效向量 |

---

## Phase 1 验证报告（2026-03-12 ~ 03-19）

### 迭代过程

| 版本 | 维度 | 坐标轴 | 样本数 | 整体准确率 | LONG | SHORT | 相似度范围 |
|------|------|--------|--------|-----------|------|-------|-----------|
| v1 | 768 | 无 | 1833 | 54.3% | 52.2% | 58.3% | 0.981~0.994 |
| v2 | 3072 | 无 | 904 | 47.8% | 55.0% | 0.0% | 0.980~0.993 |
| v3 | 3072 | 有 | 1004 | 56.4% | 65.5% | 30.0% | 0.959~0.984 |
| **v4** | **3072** | **有** | **4480** | **55.6%** | **54.8%** | **57.1%** | **0.954~0.984** |

### 关键发现

1. **坐标轴至关重要** — Gemini Embedding 2 是多模态模型，同时读图片中的视觉和文字。去掉坐标轴（axisoff=True）让模型损失了价格、时间、成交量等语义信息，相似度区分度极低（0.98~0.99）。加回坐标轴后区分度扩展到 0.95~0.98。

2. **维度影响不大** — 768维和3072维在相同条件下差异不显著。信息瓶颈在图片内容本身，不在向量维度。

3. **55.6% 的天花板** — 全量4480样本、3072维、带坐标轴的最优配置下，准确率为55.6%。比随机(50%)好，但未达到60%的实战门槛。

4. **LONG/SHORT 基本均衡** — 覆盖完整牛熊周期（2020-2026）后，LONG 54.8%、SHORT 57.1%，没有严重偏差。

5. **API 成本极低** — 4512张图片 Embedding 成本 < $1，Qdrant 免费。

### 验收结论

按原方案的判断标准：
- Hit Rate > 60% → 有价值，继续 ❌
- Hit Rate 55-60% → 需要增加过滤条件 ⚠️ **← 当前位置**
- Hit Rate < 55% → 纯形态不够，考虑其他维度 ❌

**结论：纯形态 Embedding 作为独立交易信号不够用，但作为辅助过滤条件有价值。**

---

## 实战运用方案

### 定位：辅助信号，不是主信号

55.6% 的准确率不适合独立决策，但可以作为**交易过滤器**——当你已经有交易想法时，用它来确认或否决。

### 方案 A：信号共振（推荐）

将 VectorTrader 信号与你现有的交易体系结合：

```
你的主信号（技术分析/BigBrother V6）
    ↓ 产生交易想法
VectorTrader 相似形态检索
    ↓
如果 Top-10 历史形态中 >= 7个方向一致 → 加分，执行
如果 Top-10 方向分散（4-6个）→ 中性，按原计划
如果 Top-10 中 >= 7个方向相反 → 减分，谨慎或放弃
```

**不用它做决策，用它做确认。** 就像出门前看天气预报——不会因为预报晴天就出门，但如果你本来就想出门，晴天预报会让你更放心。

### 方案 B：极端信号过滤

只在信号极度一致时才参考：

```python
# 只关注 Top-10 中 >= 9个方向一致的极端情况
if confidence >= 0.9 and avg_similarity >= 0.98:
    # 这种"高度共识"的情况值得关注
    send_alert(symbol, direction, confidence)
```

这种极端信号出现频率低（约10%的检索会触发），但方向性可能更强。

### 方案 C：实时监控推送

每4小时自动检索一次，只在极端信号时推送：

```bash
# 定时任务：每4小时跑一次
python scripts/search_pattern.py --symbol BTCUSDT --interval 4h
```

推送到微信/Telegram，配合人工判断。

---

## 下一步优化方向

| 方向 | 难度 | 预期收益 | 说明 |
|------|------|---------|------|
| **多品种扩展** | 低 | 中 | 加入 ETH、螺纹钢，扩大形态库 |
| **多周期叠加** | 低 | 中 | 同时看 1h/4h/1d 三个级别的形态共振 |
| **市场环境标签** | 中 | 高 | 区分趋势/震荡行情，分环境统计胜率 |
| **图片增强** | 中 | 未知 | 加入均线、布林带等技术指标到图片中 |
| **文本+图片混合Embedding** | 中 | 高 | 图片+K线数值描述一起Embedding |
| **专用模型微调** | 高 | 可能高 | 用金融图表数据微调Embedding模型 |

---

## 快速开始

```bash
# 1. 安装依赖
pip install -r requirements.txt

# 2. 配置 API Key（付费层，无每日限额困扰）
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

## 项目结构

```
VectorTrader/
├── config/
│   ├── __init__.py
│   └── settings.py          # 全局配置
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

---

## 花费总结

| 项目 | 费用 |
|------|------|
| Gemini Embedding 2 (4512张) | < $1 |
| Qdrant | 免费（本地Docker） |
| Binance 数据 | 免费 |
| 服务器 | 无（本地运行） |
| **总计** | **< $1** |
| **时间** | **7天**（主要是等免费层API配额） |
