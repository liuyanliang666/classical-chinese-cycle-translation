import sys
from pathlib import Path

# 让 pytest 能直接 import ccnlp（src 布局），无需手动设置 PYTHONPATH。
sys.path.insert(0, str(Path(__file__).resolve().parent / "src"))
