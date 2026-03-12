"""VectorTrader 全局配置"""
import os
from pathlib import Path

# 项目根目录
ROOT = Path(__file__).resolve().parents[1]

# 加载 .env 文件
_env_path = ROOT / '.env'
if _env_path.exists():
    with open(_env_path) as _f:
        for _line in _f:
            _line = _line.strip()
            if _line and not _line.startswith('#') and '=' in _line:
                _key, _val = _line.split('=', 1)
                os.environ.setdefault(_key.strip(), _val.strip())

# 数据目录
DATA_DIR = ROOT / "data"
CHARTS_DIR = ROOT / "charts"

# Gemini API
GEMINI_API_KEY = os.environ.get("GEMINI_API_KEY", "")

# Binance (公开数据不需要key，拉历史K线免认证)
BINANCE_API_KEY = os.environ.get("BINANCE_API_KEY", "")
BINANCE_API_SECRET = os.environ.get("BINANCE_API_SECRET", "")

# Qdrant
QDRANT_HOST = "localhost"
QDRANT_PORT = 6333
COLLECTION_NAME = "kline_patterns"

# 形态参数
WINDOW_SIZE = 20       # 每个形态截取的K线根数（20根4h = 约3.3天）
STEP_SIZE = 3          # 滑动步长（3根4h = 12小时）
DEFAULT_INTERVAL = '4h'
EMBEDDING_DIM = 768    # 向量维度（从3072截断）

# 交易品种
SYMBOLS = ["BTCUSDT"]  # Phase 1 只做BTC
DEFAULT_SYMBOL = "BTCUSDT"

# 信号阈值
SIMILARITY_THRESHOLD = 0.90   # 最低相似度
CONFIDENCE_THRESHOLD = 0.70   # 最低胜率（多）/ 最高胜率（空 < 0.30）
TOP_K = 20                    # 检索数量
