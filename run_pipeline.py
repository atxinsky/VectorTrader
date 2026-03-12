"""
VectorTrader 一键执行管线
按顺序执行: 拉数据 → 生成图片 → Embedding → 导入Qdrant → 检索验证

用法:
  python run_pipeline.py                    # 全流程
  python run_pipeline.py --step fetch       # 只拉数据
  python run_pipeline.py --step charts      # 只生成图片
  python run_pipeline.py --step embed       # 只做Embedding
  python run_pipeline.py --step import      # 只导入Qdrant
  python run_pipeline.py --step search      # 只做检索
"""
import argparse
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent
SCRIPTS = ROOT / "scripts"


def run_step(name: str, script: str, args: list[str] = None):
    """执行一个步骤"""
    print(f"\n{'='*60}")
    print(f"  Step: {name}")
    print(f"{'='*60}\n")

    cmd = [sys.executable, str(SCRIPTS / script)]
    if args:
        cmd.extend(args)

    result = subprocess.run(cmd, cwd=str(ROOT))
    if result.returncode != 0:
        print(f"\n  {name} 失败! (exit code: {result.returncode})")
        return False
    return True


def main():
    parser = argparse.ArgumentParser(description='VectorTrader Pipeline')
    parser.add_argument('--step', default='all',
                        choices=['all', 'fetch', 'charts', 'embed', 'import', 'search'],
                        help='执行哪个步骤')
    parser.add_argument('--symbol', default='BTCUSDT')
    parser.add_argument('--interval', default='4h')
    args = parser.parse_args()

    common_args = ['--symbol', args.symbol, '--interval', args.interval]
    step = args.step

    steps = {
        'fetch':  ('1/5 拉取K线数据', 'fetch_klines.py', common_args),
        'charts': ('2/5 生成K线图片', 'generate_charts.py', common_args),
        'embed':  ('3/5 Gemini Embedding', 'embed_charts.py', common_args),
        'import': ('4/5 导入Qdrant', 'import_qdrant.py', common_args),
        'search': ('5/5 形态检索验证', 'search_pattern.py', common_args),
    }

    if step == 'all':
        for key, (name, script, step_args) in steps.items():
            if not run_step(name, script, step_args):
                print(f"\n管线在 '{key}' 步骤中断。")
                return
        print(f"\n{'='*60}")
        print(f"  全部完成!")
        print(f"{'='*60}")
    else:
        name, script, step_args = steps[step]
        run_step(name, script, step_args)


if __name__ == '__main__':
    main()
