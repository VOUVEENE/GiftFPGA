##
# @file   placement_utils.py
# @brief  布局相关的工具函数
#

import numpy as np
import logging

class PlacementUtils:
    """
    @brief 布局相关的工具函数集合
    """
    
    @staticmethod
    def calculate_initial_center(placedb):
        """
        @brief 计算初始中心位置
        @param placedb placement database
        @return initLocX, initLocY 初始中心坐标
        """
        numPins = 0
        initLocX = 0
        initLocY = 0
        
        if placedb.num_terminals > 0:
            # 使用固定引脚位置的加权平均值作为初始位置
            for nodeID in range(placedb.num_movable_nodes, placedb.num_physical_nodes):
                if hasattr(placedb, 'node2pin_map') and nodeID < len(placedb.node2pin_map):
                    for pID in placedb.node2pin_map[nodeID]:
                        initLocX += placedb.node_x[nodeID] + placedb.pin_offset_x[pID]
                        initLocY += placedb.node_y[nodeID] + placedb.pin_offset_y[pID]
                    numPins += len(placedb.node2pin_map[nodeID])
            
            if numPins > 0:
                initLocX /= numPins
                initLocY /= numPins
            else:
                initLocX = 0.5 * (placedb.xh - placedb.xl)
                initLocY = 0.5 * (placedb.yh - placedb.yl)
        else:
            # 设计没有IO引脚 - 放置在中心
            initLocX = 0.5 * (placedb.xh - placedb.xl)
            initLocY = 0.5 * (placedb.yh - placedb.yl)
        
        logging.debug(f"计算得到初始中心位置: ({initLocX:.2f}, {initLocY:.2f})")
        return initLocX, initLocY
    
    @staticmethod
    def apply_boundary_constraints(x, y, node_size_x, node_size_y, xl, yl, xh, yh):
        """
        @brief 应用边界约束
        @param x, y 当前位置
        @param node_size_x, node_size_y 节点尺寸
        @param xl, yl, xh, yh 边界
        @return 约束后的位置
        """
        half_width = node_size_x / 2
        half_height = node_size_y / 2
        
        constrained_x = max(xl + half_width, min(x, xh - half_width))
        constrained_y = max(yl + half_height, min(y, yh - half_height))
        
        return constrained_x, constrained_y