"""pytest 全局配置：让测试能 import src 包"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
