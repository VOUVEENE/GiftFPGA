#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
@file   large_network_analyzer.py
@author ZXJia
@date   2025
@brief  Large Network Analyzer for FPGA Placement
        Analyzes and classifies large networks in FPGA designs
"""

import numpy as np
import time
import logging
from collections import defaultdict, Counter
import os

class LargeNetworkAnalyzer:
    """超大网络分析器 - 独立模块
    
    分析FPGA设计中的超大网络，识别网络类型（时钟、电源、复位等），
    并提供处理策略建议
    """
    
    def __init__(self, placedb, params=None):
        """
        初始化网络分析器
        
        参数:
        placedb: 布局数据库
        params: 参数对象，包含分析配置
        """
        self.placedb = placedb
        self.params = params
        
        # 分析参数设置
        if params:
            self.analysis_threshold = getattr(params, 'net_analysis_threshold', 50)
            self.skip_threshold = getattr(params, 'net_skip_threshold', 200)
            self.enable_detailed_export = getattr(params, 'net_analysis_export_detailed', True)
            self.export_path = getattr(params, 'net_analysis_export_path', '.')
        else:
            # 默认参数
            self.analysis_threshold = 50
            self.skip_threshold = 200
            self.enable_detailed_export = True
            self.export_path = '.'
        
        logging.info(f"网络分析器初始化 - 分析阈值: {self.analysis_threshold}, 跳过阈值: {self.skip_threshold}")
    
    def analyze_large_networks(self, size_threshold=None):
        """
        分析超大网络的类型和特征
        
        参数:
        size_threshold: 网络大小阈值，超过此值的网络将被分析
        
        返回:
        large_nets: 包含所有大网络信息的列表
        """
        if size_threshold is None:
            size_threshold = self.analysis_threshold
            
        logging.info(f"开始分析网络大小 > {size_threshold} 的超大网络...")
        
        placedb = self.placedb
        large_nets = []
        net_type_stats = defaultdict(list)
        
        analysis_start_time = time.time()
        
        for net_id in range(placedb.num_nets):
            pin_start = placedb.flat_net2pin_start_map[net_id]
            pin_end = placedb.flat_net2pin_start_map[net_id + 1]
            num_pins = pin_end - pin_start
            
            if num_pins <= size_threshold:
                continue
                
            # 获取网络名称
            net_name = self.get_net_name(net_id)
            
            # 分析网络类型
            net_type = self.classify_network_type(net_id, net_name, num_pins)
            
            # 分析连接的节点类型
            node_types, connected_nodes = self.analyze_connected_nodes(net_id)
            
            # 计算网络的几何分布
            spatial_info = self.analyze_spatial_distribution(net_id)
            
            # 分析驱动器信息
            driver_info = self.analyze_driver_info(net_id)
            
            large_net_info = {
                'net_id': net_id,
                'net_name': net_name,
                'num_pins': num_pins,
                'net_type': net_type,
                'node_types': node_types,
                'connected_nodes': connected_nodes,
                'spatial_span': spatial_info,
                'driver_info': driver_info,
                'driver_count': driver_info['driver_count'],
                'sink_count': num_pins - driver_info['driver_count']
            }
            
            large_nets.append(large_net_info)
            net_type_stats[net_type].append(num_pins)
        
        analysis_time = time.time() - analysis_start_time
        logging.info(f"网络分析完成，耗时: {analysis_time:.3f}s")
        
        # 显示分析结果
        self.display_analysis_results(large_nets, net_type_stats, size_threshold)
        
        # 导出详细报告（如果启用）
        if len(large_nets) > 0 and self.enable_detailed_export:
            self.export_detailed_reports(large_nets)
        
        return large_nets
    
    def get_net_name(self, net_id):
        """获取网络名称"""
        if hasattr(self.placedb, 'net_names') and self.placedb.net_names is not None:
            if net_id < len(self.placedb.net_names):
                return self.placedb.net_names[net_id]
        return f"net_{net_id}"
    
    def get_node_name(self, node_id):
        """获取节点名称"""
        if hasattr(self.placedb, 'node_names') and self.placedb.node_names is not None:
            if node_id < len(self.placedb.node_names):
                return self.placedb.node_names[node_id]
        return f"node_{node_id}"
    
    def classify_network_type(self, net_id, net_name, num_pins):
        """根据网络名称和特征分类网络类型"""
        net_name_lower = net_name.lower()
        
        # 时钟网络识别 - 更全面的关键词
        clock_keywords = ['clk', 'clock', 'ck_', '_ck', 'clkin', 'clkout', 'clk_', 
                         'sysclk', 'refclk', 'pllclk', 'clk_div', 'clk_mux', 'bufg']
        if any(keyword in net_name_lower for keyword in clock_keywords):
            return 'CLOCK'
        
        # 电源网络识别
        power_keywords = ['vdd', 'vcc', 'vss', 'gnd', 'power', 'supply', 'avdd', 'dvdd',
                         'vccint', 'vccaux', 'vccbram', 'vcco', 'vccadc']
        if any(keyword in net_name_lower for keyword in power_keywords):
            return 'POWER'
        
        # 复位网络识别
        reset_keywords = ['rst', 'reset', 'resetn', 'res_n', 'areset', 'sreset',
                         'rst_n', 'reset_n', 'por', 'warm_reset', 'cold_reset', 'ibuf']
        if any(keyword in net_name_lower for keyword in reset_keywords):
            return 'RESET'
        
        # 使能信号识别
        enable_keywords = ['en', 'enable', '_en', 'en_', 'ce', 'chip_enable',
                          'write_en', 'read_en', 'clk_en', 'clock_enable']
        if any(keyword in net_name_lower for keyword in enable_keywords):
            return 'ENABLE'
        
        # 控制信号识别
        control_keywords = ['ctrl', 'control', 'sel', 'select', 'mux_sel',
                           'addr_sel', 'bank_sel', 'mode', 'config']
        if any(keyword in net_name_lower for keyword in control_keywords):
            return 'CONTROL'
        
        # 总线信号识别
        bus_keywords = ['data', 'addr', 'address', 'bus', 'dq', 'din', 'dout',
                       'a[', 'd[', 'addr[', 'data[']
        if any(keyword in net_name_lower for keyword in bus_keywords):
            return 'BUS'
        
        # 根据引脚数量进一步分类
        if num_pins > 1000:
            return 'ULTRA_LARGE'  # 超大网络，可能是全局信号
        elif num_pins > 500:
            return 'GLOBAL'  # 全局信号
        elif num_pins > 200:
            return 'LARGE_FANOUT'  # 大扇出
        elif num_pins > 100:
            return 'MEDIUM_FANOUT'  # 中等扇出
        
        return 'UNKNOWN'
    
    def analyze_connected_nodes(self, net_id):
        """分析连接到网络的节点类型和具体节点名称"""
        placedb = self.placedb
        pin_start = placedb.flat_net2pin_start_map[net_id]
        pin_end = placedb.flat_net2pin_start_map[net_id + 1]
        
        node_type_count = Counter()
        connected_nodes = []  # 存储连接的节点信息
        
        for pin_idx in range(pin_start, pin_end):
            flat_pin_id = placedb.flat_net2pin_map[pin_idx]
            node_id = placedb.pin2node_map[flat_pin_id]
            
            if node_id < placedb.num_physical_nodes:
                node_name = self.get_node_name(node_id)
                
                # 分析节点类型
                if node_id >= placedb.num_movable_nodes:
                    node_type = 'IO'
                    node_type_count['IO'] += 1
                elif hasattr(placedb, 'lut_mask') and placedb.lut_mask[node_id]:
                    node_type = 'LUT'
                    node_type_count['LUT'] += 1
                elif hasattr(placedb, 'flop_mask') and placedb.flop_mask[node_id]:
                    node_type = 'FF'
                    node_type_count['FF'] += 1
                elif hasattr(placedb, 'dsp_mask') and placedb.dsp_mask[node_id]:
                    node_type = 'DSP'
                    node_type_count['DSP'] += 1
                elif hasattr(placedb, 'ram_mask') and placedb.ram_mask[node_id]:
                    node_type = 'RAM'
                    node_type_count['RAM'] += 1
                else:
                    node_type = 'OTHER'
                    node_type_count['OTHER'] += 1
                
                connected_nodes.append({
                    'node_id': node_id,
                    'node_name': node_name,
                    'node_type': node_type
                })
        
        return dict(node_type_count), connected_nodes
    
    def analyze_driver_info(self, net_id):
        """分析网络的驱动器信息"""
        # 简化实现：大多数网络只有一个驱动器
        # 在实际实现中，可以根据引脚方向信息来精确判断
        driver_count = 1
        
        # 特殊情况：双向信号或多驱动网络
        net_name = self.get_net_name(net_id)
        if 'bidir' in net_name.lower() or 'inout' in net_name.lower():
            driver_count = 2  # 双向信号
        
        return {
            'driver_count': driver_count,
            'has_tristate': False,  # 可以通过分析引脚类型来确定
            'is_bidirectional': 'bidir' in net_name.lower()
        }
    
    def analyze_spatial_distribution(self, net_id):
        """分析网络的空间分布"""
        placedb = self.placedb
        pin_start = placedb.flat_net2pin_start_map[net_id]
        pin_end = placedb.flat_net2pin_start_map[net_id + 1]
        
        if not hasattr(placedb, 'node_x') or not hasattr(placedb, 'node_y'):
            return {'span_x': 0, 'span_y': 0, 'distribution': 'UNKNOWN'}
        
        x_coords = []
        y_coords = []
        
        for pin_idx in range(pin_start, pin_end):
            flat_pin_id = placedb.flat_net2pin_map[pin_idx]
            node_id = placedb.pin2node_map[flat_pin_id]
            
            if node_id < placedb.num_physical_nodes:
                x_coords.append(placedb.node_x[node_id])
                y_coords.append(placedb.node_y[node_id])
        
        if not x_coords:
            return {'span_x': 0, 'span_y': 0, 'distribution': 'EMPTY'}
        
        span_x = max(x_coords) - min(x_coords)
        span_y = max(y_coords) - min(y_coords)
        
        # 判断分布类型
        chip_width = placedb.xh - placedb.xl
        chip_height = placedb.yh - placedb.yl
        
        if span_x > 0.8 * chip_width or span_y > 0.8 * chip_height:
            distribution = 'GLOBAL'
        elif span_x > 0.3 * chip_width or span_y > 0.3 * chip_height:
            distribution = 'REGIONAL'
        else:
            distribution = 'LOCAL'
        
        return {
            'span_x': span_x,
            'span_y': span_y,
            'distribution': distribution,
            'center_x': np.mean(x_coords),
            'center_y': np.mean(y_coords)
        }
    
    def display_analysis_results(self, large_nets, net_type_stats, size_threshold):
        """显示分析结果"""
        if not large_nets:
            logging.info(f"没有发现大小 > {size_threshold} 的超大网络")
            return
        
        logging.info(f"\n=== 超大网络分析报告 (>{size_threshold} 引脚) ===")
        logging.info(f"总共发现 {len(large_nets)} 个超大网络")
        
        # 按类型统计
        logging.info(f"\n--- 网络类型分布 ---")
        for net_type, sizes in net_type_stats.items():
            count = len(sizes)
            avg_size = np.mean(sizes)
            max_size = max(sizes)
            min_size = min(sizes)
            logging.info(f"{net_type:15}: {count:3}个, 平均{avg_size:6.1f}引脚, 范围[{min_size:4}-{max_size:4}]")
        
        # 显示最大的几个网络详情
        logging.info(f"\n--- 最大的10个网络详情 ---")
        large_nets_sorted = sorted(large_nets, key=lambda x: x['num_pins'], reverse=True)
        
        for i, net_info in enumerate(large_nets_sorted[:10]):
            logging.info(f"{i+1:2}. 网络: {net_info['net_name'][:40]:40} "
                        f"引脚数: {net_info['num_pins']:4} "
                        f"类型: {net_info['net_type']:15} "
                        f"分布: {net_info['spatial_span']['distribution']:8}")
            
            # 显示连接的节点类型分布
            node_types = net_info['node_types']
            if node_types:
                type_str = ", ".join([f"{k}:{v}" for k, v in node_types.items()])
                logging.info(f"     节点类型: {type_str}")
            
            # 显示部分连接的具体节点名称
            connected_nodes = net_info['connected_nodes']
            if connected_nodes and len(connected_nodes) <= 20:
                # 如果节点数不多，显示所有节点
                node_names = [f"{node['node_name']}({node['node_type']})" for node in connected_nodes[:10]]
                if len(connected_nodes) > 10:
                    node_names.append(f"...等{len(connected_nodes)}个节点")
                logging.info(f"     连接节点: {', '.join(node_names)}")
            elif connected_nodes:
                # 如果节点很多，只显示前几个代表性节点
                sample_nodes = connected_nodes[:5]
                node_names = [f"{node['node_name']}({node['node_type']})" for node in sample_nodes]
                logging.info(f"     连接节点示例: {', '.join(node_names)}...等{len(connected_nodes)}个节点")
        
        # 给出处理建议
        logging.info(f"\n--- 处理建议 ---")
        for net_type, sizes in net_type_stats.items():
            max_size = max(sizes)
            count = len(sizes)
            if net_type in ['CLOCK', 'POWER', 'RESET']:
                logging.info(f"{net_type:15}: 建议跳过 ({count}个网络, 最大{max_size}引脚)")
            elif net_type in ['ULTRA_LARGE']:
                logging.info(f"{net_type:15}: 建议跳过或极简处理 ({count}个网络, 最大{max_size}引脚)")
            elif max_size > 500:
                logging.info(f"{net_type:15}: 建议使用星形模型 ({count}个网络, 最大{max_size}引脚)")
            elif max_size > 100:
                logging.info(f"{net_type:15}: 建议使用混合模型 ({count}个网络, 最大{max_size}引脚)")
            else:
                logging.info(f"{net_type:15}: 可使用标准clique模型 ({count}个网络, 最大{max_size}引脚)")
    
    def export_detailed_reports(self, large_nets):
        """导出详细的网络分析报告到文件"""
        try:
            # 创建输出目录
            os.makedirs(self.export_path, exist_ok=True)
            
            # 主报告文件
            main_report_file = os.path.join(self.export_path, "large_networks_analysis.txt")
            self.export_main_report(large_nets, main_report_file)
            
            # 按类型分类的报告
            type_report_file = os.path.join(self.export_path, "networks_by_type.txt")
            self.export_type_classification_report(large_nets, type_report_file)
            
            # CSV格式的简化报告
            csv_report_file = os.path.join(self.export_path, "large_networks_summary.csv")
            self.export_csv_summary(large_nets, csv_report_file)
            
            logging.info(f"详细报告已导出到目录: {self.export_path}")
            
        except Exception as e:
            logging.error(f"导出报告失败: {e}")
    
    def export_main_report(self, large_nets, output_file):
        """导出主要分析报告"""
        with open(output_file, 'w', encoding='utf-8') as f:
            f.write("# 超大网络详细分析报告\n")
            f.write("=" * 80 + "\n\n")
            f.write(f"分析时间: {time.strftime('%Y-%m-%d %H:%M:%S')}\n")
            f.write(f"分析的网络数量: {len(large_nets)}\n")
            f.write(f"分析阈值: {self.analysis_threshold} 引脚\n\n")
            
            # 按网络大小排序
            large_nets_sorted = sorted(large_nets, key=lambda x: x['num_pins'], reverse=True)
            
            for i, net_info in enumerate(large_nets_sorted):
                f.write(f"## 网络 {i+1}: {net_info['net_name']}\n")
                f.write(f"- 网络ID: {net_info['net_id']}\n")
                f.write(f"- 引脚数: {net_info['num_pins']}\n")
                f.write(f"- 网络类型: {net_info['net_type']}\n")
                f.write(f"- 空间分布: {net_info['spatial_span']['distribution']}\n")
                f.write(f"- 驱动器数: {net_info['driver_count']}\n")
                f.write(f"- 接收器数: {net_info['sink_count']}\n")
                
                # 节点类型统计
                node_types = net_info['node_types']
                if node_types:
                    f.write("- 节点类型分布:\n")
                    for node_type, count in node_types.items():
                        f.write(f"  - {node_type}: {count}个\n")
                
                # 连接的节点详情
                connected_nodes = net_info['connected_nodes']
                if connected_nodes:
                    f.write("- 连接的节点:\n")
                    for node in connected_nodes[:30]:  # 最多显示30个节点
                        f.write(f"  - {node['node_name']} ({node['node_type']})\n")
                    if len(connected_nodes) > 30:
                        f.write(f"  - ... 等共{len(connected_nodes)}个节点\n")
                
                f.write("\n")
    
    def export_type_classification_report(self, large_nets, output_file):
        """导出按类型分类的报告"""
        with open(output_file, 'w', encoding='utf-8') as f:
            f.write("# 网络类型分类报告\n")
            f.write("=" * 60 + "\n\n")
            
            # 按类型分组
            nets_by_type = defaultdict(list)
            for net in large_nets:
                nets_by_type[net['net_type']].append(net)
            
            for net_type, nets in nets_by_type.items():
                f.write(f"## {net_type} 类型网络 ({len(nets)}个)\n\n")
                
                nets_sorted = sorted(nets, key=lambda x: x['num_pins'], reverse=True)
                for net in nets_sorted:
                    f.write(f"- {net['net_name']} ({net['num_pins']}引脚)\n")
                f.write("\n")
    
    def export_csv_summary(self, large_nets, output_file):
        """导出CSV格式的摘要报告"""
        with open(output_file, 'w', encoding='utf-8') as f:
            # CSV头部
            f.write("NetID,NetName,PinCount,NetType,Distribution,DriverCount,SinkCount,LUT,FF,DSP,RAM,IO,OTHER\n")
            
            # 数据行
            for net in large_nets:
                node_types = net['node_types']
                f.write(f"{net['net_id']},{net['net_name']},{net['num_pins']},{net['net_type']},"
                       f"{net['spatial_span']['distribution']},{net['driver_count']},{net['sink_count']},"
                       f"{node_types.get('LUT', 0)},{node_types.get('FF', 0)},{node_types.get('DSP', 0)},"
                       f"{node_types.get('RAM', 0)},{node_types.get('IO', 0)},{node_types.get('OTHER', 0)}\n")
    
    def get_processing_strategy(self, net_info):
        """根据网络信息返回推荐的处理策略"""
        net_type = net_info['net_type']
        num_pins = net_info['num_pins']
        
        if net_type in ['CLOCK', 'POWER', 'RESET']:
            return 'SKIP'
        elif net_type in ['ULTRA_LARGE'] or num_pins > self.skip_threshold:
            return 'SKIP'
        elif num_pins > 100:
            return 'STAR'
        elif num_pins > 50:
            return 'HYBRID'
        else:
            return 'CLIQUE'