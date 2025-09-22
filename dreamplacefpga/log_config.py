import logging
import sys

def setup_logging():
    """统一的日志配置"""
    # 设置根logger
    root_logger = logging.getLogger()
    root_logger.setLevel(logging.INFO)
    
    # 清>handlers
    root_logger.handlers.clear()
    
    # 创建新的handler
    handler = logging.StreamHandler(sys.stdout)
    formatter = logging.Formatter('[%(levelname)-7s] %(name)s - %(message)s')
    handler.setFormatter(formatter)
    root_logger.addHandler(handler)
    
    # 设置特定logger
    logging.getLogger('gift_init_placer').setLevel(logging.INFO)
    logging.getLogger('DREAMPlaceFPGA').setLevel(logging.INFO)
