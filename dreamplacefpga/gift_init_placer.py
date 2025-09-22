# gift_init_placer.py
import numpy as np
import scipy.sparse as sp
import matplotlib.pyplot as plt
from collections import defaultdict
import time
import os
import logging
from placement_utils import PlacementUtils

# 导入独立的网络分析器
try:
    from large_network_analyzer import LargeNetworkAnalyzer
    ANALYZER_AVAILABLE = True
except ImportError:
    logging.warning("large_network_analyzer.py 未找到，网络分析功能将被禁用")
    ANALYZER_AVAILABLE = False

class GiFtFPGAPlacer:
    """
    基于图信号处理的FPGA布局加速器 - 修复版本
    基于论文: The Power of Graph Signal Processing for Chip Placement Acceleration
    
    主要优化：
    1. 使用迭代方式计算矩阵幂，避免稀疏度退化
    2. 预构建归一化邻接矩阵，避免重复计算
    3. 优化内存使用和计算效率
    4. 完整的错误处理机制
    """
    
    def __init__(self, placedb, params):
        """
        初始化GiFt布局器
        
        参数:
        placedb: 布局数据库
        params: 参数设置
        """
        self.placedb = placedb
        self.params = params
        
        # 从参数中获取GiFt算法参数
        self.alpha0 = getattr(params, 'gift_alpha0', 0.1)  # 高通滤波器权重
        self.alpha1 = getattr(params, 'gift_alpha1', 0.7)  # 中通滤波器权重
        self.alpha2 = getattr(params, 'gift_alpha2', 0.2)  # 低通滤波器权重
        
        # 优化参数
        self.max_net_size = getattr(params, 'gift_max_net_size', 1000)
        self.use_star_model = getattr(params, 'gift_use_star_model', True)
        self.optimize_large_nets = getattr(params, 'gift_optimize_large_nets', True)
        
        # sigma参数（控制归一化强度）
        self.sigma_2 = getattr(params, 'gift_sigma_2', 1.5)
        self.sigma_4 = getattr(params, 'gift_sigma_4', 3.0)
        
        # 约束控制参数
        self.enable_boundary_constraints = getattr(params, 'gift_enable_boundary_constraints', True)
        self.enable_resource_constraints = getattr(params, 'gift_enable_resource_constraints', True)
        
        # 网络分析参数
        self.enable_network_analysis = getattr(params, 'enable_network_analysis', False) and ANALYZER_AVAILABLE
        self.analysis_threshold = getattr(params, 'net_analysis_threshold', 50)
        self.skip_threshold = getattr(params, 'net_skip_threshold', 200)
        
        # 布局区域边界
        self.xl = placedb.xl
        self.yl = placedb.yl
        self.xh = placedb.xh
        self.yh = placedb.yh
        
        # 节点映射
        self.node_to_idx = {}
        self.idx_to_node = {}
        for i in range(placedb.num_physical_nodes):
            if hasattr(placedb, 'node_names') and placedb.node_names is not None:
                if i < len(placedb.node_names):
                    node_name = placedb.node_names[i]
                else:
                    node_name = f"node_{i}"
            else:
                node_name = f"node_{i}"
            self.node_to_idx[node_name] = i
            self.idx_to_node[i] = node_name
        
        # 用于保存位置
        self.initial_positions = None
        self.optimized_positions = None
        
        # 预构建的归一化邻接矩阵（避免重复计算）
        self.A_tilde_2 = None
        self.A_tilde_4 = None
        self.adjacency_built = False
        
        # 资源区域定义
        self.resource_regions = {}
        # 从placedb导入资源区域信息
        self.import_resource_regions()
        
        # 初始化网络分析器（如果启用）
        if self.enable_network_analysis:
            try:
                self.network_analyzer = LargeNetworkAnalyzer(placedb, params)
                logging.info("网络分析器已启用")
            except Exception as e:
                logging.error(f"网络分析器初始化失败: {e}")
                self.network_analyzer = None
                self.enable_network_analysis = False
        else:
            self.network_analyzer = None
        
        logging.info(f"GiFt优化版本初始化完成 - 边界约束: {'启用' if self.enable_boundary_constraints else '禁用'}, "
                    f"资源约束: {'启用' if self.enable_resource_constraints else '禁用'}, "
                    f"网络分析: {'启用' if self.enable_network_analysis else '禁用'}")

    def set_preset_center(self, center_x, center_y):
        """
        @brief 设置预设的中心位置，避免重复计算
        @param center_x 中心X坐标
        @param center_y 中心Y坐标
        """
        self._preset_center = (center_x, center_y)
        logging.info(f"GiFt使用预设中心位置: ({center_x:.2f}, {center_y:.2f})")
    
    def import_resource_regions(self):
        """从placedb导入资源区域信息"""
        placedb = self.placedb
        
        # 资源类型映射
        resource_types = ['LUT', 'FF', 'DSP', 'RAM']
        
        # 导入资源区域
        if hasattr(placedb, 'region_boxes') and placedb.region_boxes:
            for i, regions in enumerate(placedb.region_boxes):
                if i < 4:  # 只处理前4个区域（资源区域）
                    resource_type = resource_types[i]
                    self.resource_regions[resource_type] = []
                    
                    for region in regions:
                        self.resource_regions[resource_type].append(region)
                    
                    logging.info(f"导入资源区域 - {resource_type}: {len(regions)}个区域")
    
    def build_adjacency_matrix(self):
        """构建邻接矩阵 - 集成网络分析功能"""
        if self.adjacency_built:
            return
            
        # 如果启用网络分析，先进行分析
        large_nets = []
        if self.enable_network_analysis and self.network_analyzer:
            try:
                large_nets = self.network_analyzer.analyze_large_networks(self.analysis_threshold)
                self.adjust_processing_strategy(large_nets)
            except Exception as e:
                logging.error(f"网络分析失败: {e}")
        
        logging.info("构建优化的邻接矩阵...")
        start_time = time.time()
        
        placedb = self.placedb
        n = placedb.num_physical_nodes
        
        # 使用字典收集边，避免重复
        edge_weights = defaultdict(float)
        processing_stats = defaultdict(int)
        
        for net_id in range(placedb.num_nets):
            pin_start = placedb.flat_net2pin_start_map[net_id]
            pin_end = placedb.flat_net2pin_start_map[net_id + 1]
            num_pins = pin_end - pin_start
            
            if num_pins < 2:
                continue
            
            # 根据配置选择处理策略
            strategy = self._get_net_processing_strategy(net_id, num_pins)
            
            if strategy == 'SKIP':
                processing_stats['SKIPPED'] += 1
                continue
            elif strategy == 'STAR':
                self._add_star_model_edges(net_id, pin_start, pin_end, edge_weights, n)
                processing_stats['STAR'] += 1
            else:  # CLIQUE
                self._add_clique_model_edges(net_id, pin_start, pin_end, edge_weights, n)
                processing_stats['CLIQUE'] += 1
        
        logging.info(f"网络处理统计: {dict(processing_stats)}")
        
        # 构建矩阵
        if not edge_weights:
            A = sp.eye(n, format='csr') * 0.1
            logging.warning("没有生成任何边，使用最小单位矩阵")
        else:
            rows, cols, weights = [], [], []
            for (i, j), w in edge_weights.items():
                rows.extend([i, j])
                cols.extend([j, i])
                weights.extend([w, w])
            
            A = sp.coo_matrix(
                (weights, (rows, cols)), 
                shape=(n, n), 
                dtype=np.float32
            ).tocsr()
        
        # 预计算归一化邻接矩阵
        logging.info("计算归一化邻接矩阵...")
        try:
            self.A_tilde_2 = self.get_normalized_adjacency(A, sigma=self.sigma_2)
            self.A_tilde_4 = self.get_normalized_adjacency(A, sigma=self.sigma_4)
        except Exception as e:
            logging.error(f"计算归一化邻接矩阵时出错: {e}")
            self.A_tilde_2 = A + self.sigma_2 * sp.eye(n, format='csr')
            self.A_tilde_4 = A + self.sigma_4 * sp.eye(n, format='csr')
        
        self.adjacency_built = True
        
        build_time = time.time() - start_time
        density = A.nnz / (n * n) * 100
        
        logging.info(f"邻接矩阵构建完成，耗时: {build_time:.3f}s")
        logging.info(f"矩阵统计: {A.nnz:,} 非零元素, 密度: {density:.4f}%")
    
    def _get_net_processing_strategy(self, net_id, num_pins):
        """确定网络的处理策略"""
        # 如果有网络分析器，使用精确分类
        if self.enable_network_analysis and self.network_analyzer:
            net_name = self.network_analyzer.get_net_name(net_id)
            net_type = self.network_analyzer.classify_network_type(net_id, net_name, num_pins)
            
            if net_type in ['CLOCK', 'POWER', 'RESET'] or num_pins > self.max_net_size:
                return 'SKIP'
            elif num_pins > 100 and self.use_star_model:
                return 'STAR'
            else:
                return 'CLIQUE'
        else:
            # 使用简单的大小基础策略
            if num_pins > self.max_net_size:
                return 'SKIP'
            elif num_pins > 100 and self.use_star_model:
                return 'STAR'
            else:
                return 'CLIQUE'
    
    def _add_star_model_edges(self, net_id, pin_start, pin_end, edge_weights, n):
        """为网络添加星形模型边"""
        placedb = self.placedb
        
        # 收集有效节点
        nodes = []
        for j in range(pin_start, pin_end):
            flat_pin_id = placedb.flat_net2pin_map[j]
            node_id = placedb.pin2node_map[flat_pin_id]
            if node_id < n:
                nodes.append(node_id)
        
        if len(nodes) < 2:
            return
        
        # 选择中心节点（简单选择第一个）
        center_node = nodes[0]
        star_weight = 2.0 / len(nodes)
        
        # 连接中心到其他所有节点
        for node_id in nodes[1:]:
            edge = (min(center_node, node_id), max(center_node, node_id))
            edge_weights[edge] += star_weight
    
    def _add_clique_model_edges(self, net_id, pin_start, pin_end, edge_weights, n):
        """为网络添加clique模型边"""
        placedb = self.placedb
        num_pins = pin_end - pin_start
        weight = 2.0 / num_pins
        
        # 收集有效节点
        nodes = []
        for j in range(pin_start, pin_end):
            flat_pin_id = placedb.flat_net2pin_map[j]
            node_id = placedb.pin2node_map[flat_pin_id]
            if node_id < n:
                nodes.append(node_id)
        
        # 生成所有节点对之间的边
        for i in range(len(nodes)):
            for j in range(i + 1, len(nodes)):
                node1, node2 = nodes[i], nodes[j]
                edge = (min(node1, node2), max(node1, node2))
                edge_weights[edge] += weight
    
    def adjust_processing_strategy(self, large_nets):
        """根据分析结果调整处理策略"""
        if not large_nets or not self.network_analyzer:
            return
            
        strategy_stats = defaultdict(int)
        
        for net_info in large_nets:
            strategy = self.network_analyzer.get_processing_strategy(net_info)
            strategy_stats[strategy] += 1
        
        if strategy_stats:
            logging.info(f"\n--- 网络处理策略分配 ---")
            total = sum(strategy_stats.values())
            for strategy, count in strategy_stats.items():
                percentage = count / total * 100
                logging.info(f"{strategy:6}: {count:3}个网络 ({percentage:5.1f}%)")
    
    def get_normalized_adjacency(self, A, sigma=2):
        """
        计算归一化邻接矩阵(增加自环) - 带错误处理的版本
        """
        try:
            n = A.shape[0]
            I = sp.eye(n, format='csr')
            A_sigma = A + sigma * I
            
            # 计算度矩阵
            row_sum = np.array(A_sigma.sum(axis=1)).flatten()
            row_sum[row_sum == 0] = 1e-6  # 避免除以零
            
            # 计算D^(-1/2) * A * D^(-1/2)
            D_sqrt_inv = sp.diags(1.0 / np.sqrt(row_sum), format='csr')
            A_tilde = D_sqrt_inv @ A_sigma @ D_sqrt_inv
            
            return A_tilde
            
        except Exception as e:
            logging.error(f"归一化计算失败: {e}")
            n = A.shape[0]
            return A + sigma * sp.eye(n, format='csr')
    
    def matrix_power_iterative(self, A_tilde, positions, k):
        """
        迭代计算 A^k * positions，避免显式计算矩阵幂
        """
        result = positions.copy()
        for i in range(k):
            result[:, 0] = A_tilde.dot(result[:, 0])
            result[:, 1] = A_tilde.dot(result[:, 1])
        
        return result
    
    def initialize_positions(self):
        """初始化节点位置 - 使用统一的工具方法"""
        logging.info("初始化节点位置...")
        placedb = self.placedb
        n = placedb.num_physical_nodes
        
        # 检查是否有预设的中心位置
        if hasattr(self, '_preset_center'):
            initLocX, initLocY = self._preset_center
            logging.info(f"使用预设中心位置: ({initLocX:.2f}, {initLocY:.2f})")
        else:
            # 如果没有预设，使用统一的工具方法计算
            initLocX, initLocY = PlacementUtils.calculate_initial_center(placedb)
            logging.info(f"GiFt独立计算中心位置: ({initLocX:.2f}, {initLocY:.2f})")
        
        # 创建初始位置数组
        initial_positions = np.zeros((n, 2), dtype=np.float32)
        
        # 设置固定节点的位置（使用中心点坐标）
        for i in range(placedb.num_movable_nodes, n):
            initial_positions[i, 0] = placedb.node_x[i] + placedb.node_size_x[i] / 2
            initial_positions[i, 1] = placedb.node_y[i] + placedb.node_size_y[i] / 2
        
        scale = min(placedb.xh - placedb.xl, placedb.yh - placedb.yl) * 0.001

        # 使用与BasicPlace一致的随机种子
        random_seed = getattr(self.params, 'random_seed', 0)
        random_gen = PlacementUtils.create_random_generator(random_seed)

        for i in range(placedb.num_movable_nodes):
            initial_positions[i, 0] = initLocX + random_gen.normal(0, scale)
            initial_positions[i, 1] = initLocY + random_gen.normal(0, scale)
            
            # 应用边界约束（如果启用）
            if self.enable_boundary_constraints:
                initial_positions[i, 0], initial_positions[i, 1] = PlacementUtils.apply_boundary_constraints(
                    initial_positions[i, 0], initial_positions[i, 1],
                    placedb.node_size_x[i], placedb.node_size_y[i],
                    placedb.xl, placedb.yl, placedb.xh, placedb.yh
                )
        
        self.initial_positions = initial_positions
        return initial_positions
    
    def apply_gift(self, initial_positions):
        """应用GiFt算法计算优化的位置"""
        logging.info(f"应用GiFt滤波器... (alpha0={self.alpha0}, alpha1={self.alpha1}, alpha2={self.alpha2})")
        
        # 确保邻接矩阵已构建
        self.build_adjacency_matrix()
        
        start_time = time.time()
        
        # 使用迭代方式计算三种滤波器的结果
        logging.info("计算高通滤波器 (A_tilde_2^2)...")
        pos_high = self.matrix_power_iterative(self.A_tilde_2, initial_positions, 2)
        
        logging.info("计算中通滤波器 (A_tilde_4^2)...")
        pos_mid = self.matrix_power_iterative(self.A_tilde_4, initial_positions, 2)
        
        logging.info("计算低通滤波器 (A_tilde_4^4)...")
        pos_low = self.matrix_power_iterative(self.A_tilde_4, initial_positions, 4)
        
        # 按照GiFt论文的公式组合结果
        optimized_positions = (
            self.alpha0 * pos_high +
            self.alpha1 * pos_mid + 
            self.alpha2 * pos_low
        )
        
        # 确保固定节点位置不变
        placedb = self.placedb
        num_movable = placedb.num_movable_nodes
        optimized_positions[num_movable:] = initial_positions[num_movable:]
        
        logging.info(f"GiFt滤波器计算完成，耗时: {time.time() - start_time:.3f}s")
        
        return optimized_positions
    
    def apply_placement_constraints(self, positions):
        """应用布局范围约束"""
        logging.info(f"应用布局约束... (边界约束: {'启用' if self.enable_boundary_constraints else '禁用'}, "
                    f"资源约束: {'启用' if self.enable_resource_constraints else '禁用'})")
        placedb = self.placedb
        constrained_positions = positions.copy()
        
        # 为可移动节点应用约束
        for i in range(placedb.num_movable_nodes):
            half_width = placedb.node_size_x[i] / 2
            half_height = placedb.node_size_y[i] / 2
            
            # 获取中心点坐标
            center_x = positions[i, 0]
            center_y = positions[i, 1]
            
            # 应用边界约束（如果启用）
            if self.enable_boundary_constraints:
                center_x = max(placedb.xl + half_width, min(center_x, placedb.xh - half_width))
                center_y = max(placedb.yl + half_height, min(center_y, placedb.yh - half_height))
            
            # 应用资源区域约束（如果启用且有资源区域信息）
            if self.enable_resource_constraints and self.resource_regions:
                resource_type = self._get_node_resource_type(i)
                if resource_type and resource_type in self.resource_regions:
                    center_x, center_y = self._apply_resource_constraints(
                        center_x, center_y, half_width, half_height, resource_type)
            
            constrained_positions[i] = (center_x, center_y)
        
        return constrained_positions
    
    def _get_node_resource_type(self, node_id):
        """获取节点的资源类型"""
        placedb = self.placedb
        
        if hasattr(placedb, 'lut_mask') and placedb.lut_mask[node_id]:
            return 'LUT'
        elif hasattr(placedb, 'flop_mask') and placedb.flop_mask[node_id]:
            return 'FF'
        elif hasattr(placedb, 'dsp_mask') and placedb.dsp_mask[node_id]:
            return 'DSP'
        elif hasattr(placedb, 'ram_mask') and placedb.ram_mask[node_id]:
            return 'RAM'
        
        return None
    
    def _apply_resource_constraints(self, center_x, center_y, half_width, half_height, resource_type):
        """应用资源区域约束"""
        regions = self.resource_regions[resource_type]
        
        # 检查是否已在某个区域内
        for region in regions:
            if (region[0] + half_width <= center_x <= region[2] - half_width and
                region[1] + half_height <= center_y <= region[3] - half_height):
                return center_x, center_y  # 已在区域内，不需要调整
        
        # 找到最近的区域并投影
        min_dist = float('inf')
        best_pos = (center_x, center_y)
        
        for region in regions:
            x_proj = max(region[0] + half_width, 
                        min(center_x, region[2] - half_width))
            y_proj = max(region[1] + half_height, 
                        min(center_y, region[3] - half_height))
            
            dist = np.sqrt((center_x - x_proj)**2 + (center_y - y_proj)**2)
            
            if dist < min_dist:
                min_dist = dist
                best_pos = (x_proj, y_proj)
        
        return best_pos
    
    def optimize_placement(self):
        """执行GiFt优化过程"""
        total_start = time.time()
        
        # 初始化位置
        if self.initial_positions is None:
            self.initialize_positions()
        
        # 应用GiFt算法
        self.optimized_positions = self.apply_gift(self.initial_positions)
        
        # 应用布局约束
        self.optimized_positions = self.apply_placement_constraints(self.optimized_positions)
        
        # 计算初始和优化后的HPWL
        initial_wl = self.calculate_hpwl(self.initial_positions)
        optimized_wl = self.calculate_hpwl(self.optimized_positions)
        
        total_time = time.time() - total_start
        
        logging.info(f"GiFt优化完成，总耗时: {total_time:.3f}s")
        logging.info(f"初始HPWL: {initial_wl:.2f}，优化后HPWL: {optimized_wl:.2f}")
        if optimized_wl < initial_wl:
            improvement = (initial_wl - optimized_wl) / initial_wl * 100
            logging.info(f"GiFt优化有效，HPWL减少了 {improvement:.2f}%")
        else:
            degradation = (optimized_wl - initial_wl) / initial_wl * 100
            logging.warning(f"GiFt优化后HPWL增加了 {degradation:.2f}%，可能需要调整参数")
        
        return self.optimized_positions
    
    def get_dreamplace_positions(self):
        """获取DREAMPlace格式的节点位置数组"""
        if self.optimized_positions is None:
            self.optimize_placement()
        
        placedb = self.placedb
        pos = np.zeros(placedb.num_nodes * 2, dtype=placedb.dtype)
        
        # 将所有节点的现有位置复制到pos数组
        pos[0:placedb.num_physical_nodes] = placedb.node_x
        pos[placedb.num_nodes:placedb.num_nodes+placedb.num_physical_nodes] = placedb.node_y
        
        # 更新可移动节点的位置（转换为左下角坐标）
        for i in range(placedb.num_movable_nodes):
            # 中心点坐标
            center_x = self.optimized_positions[i, 0]
            center_y = self.optimized_positions[i, 1]
            
            # 转换为左下角坐标
            pos[i] = center_x - placedb.node_size_x[i] / 2
            pos[placedb.num_nodes + i] = center_y - placedb.node_size_y[i] / 2
        
        return pos
    
    def calculate_hpwl(self, positions):
        """
        计算半周长线长
        
        参数:
        positions: 节点位置（中心点坐标）
        
        返回:
        hpwl: 总线长
        """
        placedb = self.placedb
        total_wl = 0
        
        for net_id in range(placedb.num_nets):
            # 获取网络中的节点
            pin_start = placedb.flat_net2pin_start_map[net_id]
            pin_end = placedb.flat_net2pin_start_map[net_id + 1]
            
            # 如果网络只有一个节点则跳过
            if pin_end - pin_start < 2:
                continue
            
            # 计算网络的边界框
            x_min = float('inf')
            x_max = float('-inf')
            y_min = float('inf')
            y_max = float('-inf')
            
            for pin_id in range(pin_start, pin_end):
                flat_pin_id = placedb.flat_net2pin_map[pin_id]
                node_id = placedb.pin2node_map[flat_pin_id]
                
                # 计算引脚位置（考虑引脚偏移）
                if node_id < placedb.num_physical_nodes:
                    node_center_x = positions[node_id, 0]
                    node_center_y = positions[node_id, 1]
                    pin_offset_x = placedb.pin_offset_x[flat_pin_id]
                    pin_offset_y = placedb.pin_offset_y[flat_pin_id]
                    
                    # 相对中心的偏移
                    pin_x = node_center_x + pin_offset_x - placedb.node_size_x[node_id] / 2
                    pin_y = node_center_y + pin_offset_y - placedb.node_size_y[node_id] / 2
                    
                    # 更新边界框
                    x_min = min(x_min, pin_x)
                    x_max = max(x_max, pin_x)
                    y_min = min(y_min, pin_y)
                    y_max = max(y_max, pin_y)
            
            # 半周长线长 = (x_max - x_min) + (y_max - y_min)
            if x_max > x_min and y_max > y_min:
                net_wl = (x_max - x_min) + (y_max - y_min)
                total_wl += net_wl
        
        return total_wl
    
    def visualize_placement(self, output_file=None, show_nets=False):
        """
        @brief 可视化当前优化后的布局结果 - 重构版本
        """
        if self.optimized_positions is None:
            logging.warning("没有优化位置数据，无法进行可视化")
            return
        
        # 调用统一的可视化方法
        self.visualize_placement_with_positions(
            self.optimized_positions, 
            output_file, 
            title="GiFt Placement Result", 
            show_nets=show_nets
        )

    def visualize_placement_with_positions(self, positions, output_file=None, title="GiFt Placement", show_nets=False):
        """
        @brief 统一的可视化方法 - 支持任意位置数组
        @param positions 要可视化的位置数组 (N, 2)
        @param output_file 输出文件路径
        @param title 图标题
        @param show_nets 是否显示网络连线
        """
        if positions is None:
            logging.warning("位置数组为空，无法进行可视化")
            return

        placedb = self.placedb
        
        plt.figure(figsize=(12, 10))
        
        # 绘制芯片边界
        plt.plot([placedb.xl, placedb.xh, placedb.xh, placedb.xl, placedb.xl], 
                [placedb.yl, placedb.yl, placedb.yh, placedb.yh, placedb.yl], 
                'k-', linewidth=2, label='chip boundary')
        
        # 绘制资源区域
        if self.enable_resource_constraints and self.resource_regions:
            colors = {'LUT': 'blue', 'FF': 'green', 'DSP': 'purple', 'RAM': 'orange'}
            for resource_type, regions in self.resource_regions.items():
                color = colors.get(resource_type, 'gray')
                for j, region in enumerate(regions):
                    x_min, y_min, x_max, y_max = region
                    label = f'{resource_type} region' if j == 0 else None
                    plt.plot([x_min, x_max, x_max, x_min, x_min], 
                            [y_min, y_min, y_max, y_max, y_min], 
                            '--', color=color, linewidth=1, alpha=0.7, label=label)
        
        # 准备绘制不同类型的节点
        lut_x, lut_y = [], []
        ff_x, ff_y = [], []
        dsp_x, dsp_y = [], []
        ram_x, ram_y = [], []
        io_x, io_y = [], []
        
        # 分类节点
        for i in range(min(placedb.num_physical_nodes, len(positions))):
            x, y = positions[i]
            
            if i >= placedb.num_movable_nodes:
                io_x.append(x)
                io_y.append(y)
            elif hasattr(placedb, 'lut_mask') and placedb.lut_mask[i]:
                lut_x.append(x)
                lut_y.append(y)
            elif hasattr(placedb, 'flop_mask') and placedb.flop_mask[i]:
                ff_x.append(x)
                ff_y.append(y)
            elif hasattr(placedb, 'dsp_mask') and placedb.dsp_mask[i]:
                dsp_x.append(x)
                dsp_y.append(y)
            elif hasattr(placedb, 'ram_mask') and placedb.ram_mask[i]:
                ram_x.append(x)
                ram_y.append(y)
        
        # 绘制各类节点
        if lut_x: plt.scatter(lut_x, lut_y, c='blue', marker='o', s=10, alpha=0.7, label='LUT')
        if ff_x: plt.scatter(ff_x, ff_y, c='green', marker='s', s=10, alpha=0.7, label='FF')
        if dsp_x: plt.scatter(dsp_x, dsp_y, c='purple', marker='^', s=20, alpha=0.8, label='DSP')
        if ram_x: plt.scatter(ram_x, ram_y, c='orange', marker='d', s=20, alpha=0.8, label='RAM')
        if io_x: plt.scatter(io_x, io_y, c='red', marker='x', s=30, label='IO')
        
        # 设置标题
        constraint_info = []
        if self.enable_boundary_constraints:
            constraint_info.append("boundary constraint")
        if self.enable_resource_constraints:
            constraint_info.append("resource constraint")
            
        constraints_str = ", ".join(constraint_info) if constraint_info else "no constraint"
        full_title = f"{title} ({constraints_str})"
        plt.title(full_title)
        
        # 移除图例中的重复项
        handles, labels = plt.gca().get_legend_handles_labels()
        by_label = dict(zip(labels, handles))
        plt.legend(by_label.values(), by_label.keys(), loc='upper right')
        
        plt.grid(True, linestyle='--', alpha=0.3)
        plt.xlabel('X')
        plt.ylabel('Y')
        
        # 保存图像
        if output_file:
            plt.savefig(output_file, dpi=300, bbox_inches='tight')
            logging.info(f"可视化结果已保存到 {output_file}")
        else:
            plt.show()
        
        plt.close()  # 关闭图形以释放内存