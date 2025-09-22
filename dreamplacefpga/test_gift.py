#!/usr/bin/env python3
import sys
import os
sys.path.insert(0, '.')

try:
    import gift_adj_cpp
    print("✓ C++扩展加载成功")
except ImportError as e:
    print("✗ C++扩展加载失败:", e)

try:
    from gift_init_placer import GiFtFPGAPlacer
    print("✓ GiFt布局器导入成功")
except ImportError as e:
    print("✗ GiFt布局器导入失败:", e)

print("测试完成")
