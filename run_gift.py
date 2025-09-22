#!/usr/bin/env python3
import sys
import logging

# 关键：设置正确的日志级别，这样GiFt的日
logging.basicConfig(level=logging.INFO, format='[%(levelname)-7s] %(name)s - %(message)s', stream=sys.stdout)

# 添加模块路径
sys.path.insert(0, 'dreamplacefpga')

if __name__ == "__main__":
    from Placer import placeFPGA
    from Params import ParamsFPGA
    
    params = ParamsFPGA()
    params.load('test/FPGA-example1.json')
    placeFPGA(params)
