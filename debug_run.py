#!/usr/bin/env python3
import sys
import os
import logging

# 设置更详细的日志
logging.basicConfig(level=logging.DEBUG, format='[%(levelname)-7s] %(name)s - %(message)s')

print("开...")

# 添加路径
sys.path.insert(0, 'dreamplacefpga')

print("导入模块...")
try:
    from Placer import *
    print("Placer模块导入成功")
except Exception as e:
    print(f"导入Placer失败: {e}")
    import traceback
    traceback.print_exc()

print("准备运行placeFPGA...")
try:
    from Params import ParamsFPGA
    params = ParamsFPGA()
    params.load('test/FPGA-example1.json')
    print("参数加载成功")
    
    print("开始布局...")
    placeFPGA(params)
    print("布局完成")
    
except Exception as e:
    print(f"运行出错: {e}")
    import traceback
    traceback.print_exc()
