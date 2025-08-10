# gift_placer_fixed.py
import numpy as np
import scipy.sparse as sp
import matplotlib.pyplot as plt
from collections import defaultdict
import time
import os
import logging

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
        
        # 约束控制参数
        self.enable_boundary_constraints = getattr(params, 'gift_enable_boundary_constraints', True)
        self.enable_resource_constraints = getattr(params, 'gift_enable_resource_constraints', True)
        
        # 布局区域边界
        self.xl = placedb.xl
        self.yl = placedb.yl
        self.xh = placedb.xh
        self.yh = placedb.yh
        
        # 节点映射
        self.node_to_idx = {}
        self.idx_to_node = {}
        for i in range(placedb.num_physical_nodes):
            node_name = placedb.node_names[i]
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
        
        logging.info(f"GiFt优化版本初始化完成 - 边界约束: {'启用' if self.enable_boundary_constraints else '禁用'}, 资源约束: {'启用' if self.enable_resource_constraints else '禁用'}")
    
    def import_resource_regions(self):
        """从placedb导入资源区域信息"""
        placedb = self.placedb
        
        # 资源类型映射
        resource_types = ['LUT', 'FF', 'DSP', 'RAM']
        
        # 导入资源区域
        for i, regions in enumerate(placedb.region_boxes):
            if i < 4:  # 只处理前4个区域（资源区域）
                resource_type = resource_types[i]
                self.resource_regions[resource_type] = []
                
                for region in regions:
                    self.resource_regions[resource_type].append(region)
                
                logging.info(f"导入资源区域 - {resource_type}: {len(regions)}个区域")
    
    def calculate_initial_center(self):
        """计算初始中心位置，与BasicPlace方法相同"""
        placedb = self.placedb
        
        numPins = 0
        initLocX = 0
        initLocY = 0
        
        if placedb.num_terminals > 0:
            numPins = 0
            # 使用固定引脚位置的平均值作为初始位置
            for nodeID in range(placedb.num_movable_nodes, placedb.num_physical_nodes):
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
        else:  # 设计没有IO引脚 - 放置在中心
            initLocX = 0.5 * (placedb.xh - placedb.xl)
            initLocY = 0.5 * (placedb.yh - placedb.yl)
        
        return initLocX, initLocY
    
    def build_adjacency_matrix(self):
        """构建节点之间的邻接矩阵 - 完整版本"""
        if self.adjacency_built:
            return
            
        logging.info("构建邻接矩阵...")
        start_time = time.time()
        
        placedb = self.placedb
        n = placedb.num_physical_nodes
        A = sp.lil_matrix((n, n), dtype=np.float32)
        
        # 添加调试信息
        logging.info(f"节点总数: {n}, 网络总数: {placedb.num_nets}")
        
        # 统计信息
        total_edges = 0
        net_size_stats = []
        
        # 使用GiFt论文的clique模型构建邻接矩阵（基于引脚）
        for net_id in range(placedb.num_nets):
            # 获取该网络的引脚范围
            pin_start = placedb.flat_net2pin_start_map[net_id]
            pin_end = placedb.flat_net2pin_start_map[net_id + 1]
            num_pins = pin_end - pin_start
            
            if num_pins < 2:
                continue  # 跳过只有一个引脚的网络
            
            net_size_stats.append(num_pins)
            
            # 使用GiFt论文的权重公式：2/M (M是引脚数)
            weight = 2.0 / num_pins
            
            # 检查net_mask过滤（如果存在）
            if hasattr(placedb, 'net_mask') and placedb.net_mask is not None:
                if not placedb.net_mask[net_id]:
                    continue
            
            # 使用clique模型：基于引脚构建边（与DREAMPlace和GiFt论文相同的方式）
            for j in range(pin_start, pin_end):
                flat_pin_id_1 = placedb.flat_net2pin_map[j]
                node_id_1 = placedb.pin2node_map[flat_pin_id_1]
                
                for k in range(j + 1, pin_end):
                    flat_pin_id_2 = placedb.flat_net2pin_map[k]
                    node_id_2 = placedb.pin2node_map[flat_pin_id_2]
                    
                    # 确保节点ID有效且不同
                    if (node_id_1 < n and node_id_2 < n and 
                        node_id_1 != node_id_2):
                        try:
                            A[node_id_1, node_id_2] += weight
                            A[node_id_2, node_id_1] += weight
                            total_edges += 1
                        except Exception as e:
                            logging.error(f"添加边时出错: net_id={net_id}, node1={node_id_1}, node2={node_id_2}, error={e}")
                            continue
        
        # 打印网络大小统计信息
        if net_size_stats:
            logging.info(f"网络大小统计:")
            logging.info(f"  平均大小: {np.mean(net_size_stats):.2f}")
            logging.info(f"  最大大小: {max(net_size_stats)}")
            logging.info(f"  大网络(>50)数量: {sum(1 for s in net_size_stats if s > 50)}")
            logging.info(f"  超大网络(>100)数量: {sum(1 for s in net_size_stats if s > 100)}")
        
        logging.info(f"构建完成，总边数: {total_edges}")
        
        # 检查矩阵是否为空
        if A.nnz == 0:
            logging.error("邻接矩阵为空！这可能表示数据有问题")
            # 创建一个最小的有效矩阵
            A = sp.eye(n, format='csr') * 0.1
            logging.warning("使用最小单位矩阵替代")
        
        A = A.tocsr()  # 转换为CSR格式以加速计算
        
        # 预计算归一化邻接矩阵
        logging.info("计算归一化邻接矩阵...")
        try:
            self.A_tilde_2 = self.get_normalized_adjacency(A, sigma=2)
            self.A_tilde_4 = self.get_normalized_adjacency(A, sigma=4)
            logging.info("归一化邻接矩阵计算成功")
        except Exception as e:
            logging.error(f"计算归一化邻接矩阵时出错: {e}")
            # 使用简单的归一化方法作为备选
            self.A_tilde_2 = A + 2 * sp.eye(n, format='csr')
            self.A_tilde_4 = A + 4 * sp.eye(n, format='csr')
            logging.warning("使用简化的归一化方法")
        
        # 确保矩阵不为None
        if self.A_tilde_2 is None or self.A_tilde_4 is None:
            logging.error("归一化邻接矩阵为None，创建默认矩阵")
            self.A_tilde_2 = sp.eye(n, format='csr')
            self.A_tilde_4 = sp.eye(n, format='csr')
        
        self.adjacency_built = True
        
        # 打印详细的稀疏度统计信息
        logging.info(f"邻接矩阵构建完成，耗时: {time.time() - start_time:.3f}s")
        logging.info(f"统计信息: 总边数={total_edges:,}")
        logging.info(f"原始邻接矩阵: {A.nnz:,} 非零元素, 密度: {A.nnz/(n*n)*100:.4f}%")
        logging.info(f"A_tilde_2: {self.A_tilde_2.nnz:,} 非零元素, 密度: {self.A_tilde_2.nnz/(n*n)*100:.4f}%")
        logging.info(f"A_tilde_4: {self.A_tilde_4.nnz:,} 非零元素, 密度: {self.A_tilde_4.nnz/(n*n)*100:.4f}%")
        
        # 如果密度仍然太高，发出警告
        density = A.nnz/(n*n)*100
        if density > 5.0:
            logging.warning(f"邻接矩阵密度过高 ({density:.2f}%)，这可能是设计特性")
            logging.warning("现代FPGA设计可能确实具有较高的连接密度")
        else:
            logging.info(f"邻接矩阵密度正常 ({density:.2f}%)")
    
    def get_normalized_adjacency(self, A, sigma=2):
        """
        计算归一化邻接矩阵(增加自环) - 带错误处理的版本
        
        参数:
        A: 邻接矩阵
        sigma: 自环权重
        
        返回:
        A_tilde: 归一化的邻接矩阵(增加自环)
        """
        try:
            # 增加自环
            n = A.shape[0]
            I = sp.eye(n, format='csr')
            A_sigma = A + sigma * I
            
            # 计算度矩阵
            row_sum = np.array(A_sigma.sum(axis=1)).flatten()
            
            # 避免除以零（孤立节点）
            row_sum[row_sum == 0] = 1.0
            
            # 计算D^(-1/2) * A * D^(-1/2)
            D_sqrt_inv = sp.diags(1.0 / np.sqrt(row_sum), format='csr')
            A_tilde = D_sqrt_inv @ A_sigma @ D_sqrt_inv
            
            # 检查结果是否有效
            if A_tilde.nnz == 0:
                logging.warning(f"归一化后矩阵为空，sigma={sigma}")
                return A_sigma  # 返回未归一化的版本
            
            return A_tilde
            
        except Exception as e:
            logging.error(f"归一化计算失败: {e}")
            # 返回简单的自环矩阵
            n = A.shape[0]
            return A + sigma * sp.eye(n, format='csr')
    
    def matrix_power_iterative(self, A_tilde, positions, k):
        """
        迭代计算 A^k * positions，避免显式计算矩阵幂
        
        参数:
        A_tilde: 归一化邻接矩阵
        positions: 初始位置 (n, 2)
        k: 幂次
        
        返回:
        result: A^k * positions 的结果
        """
        result = positions.copy()
        for i in range(k):
            # 分别计算x和y坐标
            result[:, 0] = A_tilde.dot(result[:, 0])
            result[:, 1] = A_tilde.dot(result[:, 1])
        
        return result
    
    def initialize_positions(self):
        """初始化节点位置"""
        logging.info("初始化节点位置...")
        placedb = self.placedb
        n = placedb.num_physical_nodes
        
        # 获取初始中心位置
        initLocX, initLocY = self.calculate_initial_center()
        
        # 创建初始位置数组
        initial_positions = np.zeros((n, 2), dtype=np.float32)
        
        # 设置固定节点的位置（使用中心点坐标）
        for i in range(placedb.num_movable_nodes, n):
            initial_positions[i, 0] = placedb.node_x[i] + placedb.node_size_x[i] / 2
            initial_positions[i, 1] = placedb.node_y[i] + placedb.node_size_y[i] / 2
        
        # 为可移动节点设置初始位置（围绕初始中心点随机分布）
        scale = min(placedb.xh - placedb.xl, placedb.yh - placedb.yl) * 0.001
        for i in range(placedb.num_movable_nodes):
            initial_positions[i, 0] = initLocX + np.random.normal(0, scale)
            initial_positions[i, 1] = initLocY + np.random.normal(0, scale)
            
            # 应用边界约束（如果启用）
            if self.enable_boundary_constraints:
                half_width = placedb.node_size_x[i] / 2
                half_height = placedb.node_size_y[i] / 2
                
                initial_positions[i, 0] = max(placedb.xl + half_width, 
                                          min(initial_positions[i, 0], 
                                              placedb.xh - half_width))
                initial_positions[i, 1] = max(placedb.yl + half_height, 
                                          min(initial_positions[i, 1], 
                                              placedb.yh - half_height))
        
        self.initial_positions = initial_positions
        return initial_positions
    
    def apply_gift(self, initial_positions):
        """
        应用优化的GiFt算法计算优化的位置
        
        参数:
        initial_positions: 初始位置，形状为(n, 2)的numpy数组
        
        返回:
        optimized_positions: 优化后的位置，形状为(n, 2)的numpy数组
        """
        logging.info(f"应用GiFt滤波器... (alpha0={self.alpha0}, alpha1={self.alpha1}, alpha2={self.alpha2})")
        
        # 确保邻接矩阵已构建
        self.build_adjacency_matrix()
        
        start_time = time.time()
        
        # 使用迭代方式计算三种滤波器的结果
        # 对应DREAMPlace的：
        # location_h = A_tilde_2^2 * positions (高通滤波器)
        # location_m = A_tilde_4^2 * positions (中通滤波器)
        # location_l = A_tilde_4^4 * positions (低通滤波器)
        
        logging.info("计算高通滤波器 (A_tilde_2^2)...")
        pos_high = self.matrix_power_iterative(self.A_tilde_2, initial_positions, 2)
        
        logging.info("计算中通滤波器 (A_tilde_4^2)...")
        pos_mid = self.matrix_power_iterative(self.A_tilde_4, initial_positions, 2)
        
        logging.info("计算低通滤波器 (A_tilde_4^4)...")
        pos_low = self.matrix_power_iterative(self.A_tilde_4, initial_positions, 4)
        
        # 按照GiFt论文的公式组合结果
        # 注意：这里修正了原代码中的错误，应该都用相同的坐标轴
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
        """
        应用布局范围约束，确保所有节点都在芯片边界内
        并满足资源区域约束
        """
        logging.info(f"应用布局约束... (边界约束: {'启用' if self.enable_boundary_constraints else '禁用'}, 资源约束: {'启用' if self.enable_resource_constraints else '禁用'})")
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
            
            # 应用资源区域约束（如果启用）
            if self.enable_resource_constraints:
                resource_type = None
                if placedb.lut_mask[i]:
                    resource_type = 'LUT'
                elif placedb.flop_mask[i]:
                    resource_type = 'FF'
                elif placedb.dsp_mask[i]:
                    resource_type = 'DSP'
                elif placedb.ram_mask[i]:
                    resource_type = 'RAM'
                
                # 找到对应的资源区域
                if resource_type is not None and resource_type in self.resource_regions:
                    regions = self.resource_regions[resource_type]
                    if len(regions) > 0:
                        # 检查是否已在某个区域内
                        in_region = False
                        for region in regions:
                            if (region[0] + half_width <= center_x <= region[2] - half_width and
                                region[1] + half_height <= center_y <= region[3] - half_height):
                                in_region = True
                                break
                        
                        # 如果不在区域内，找到最近的区域并投影到其中
                        if not in_region:
                            min_dist = float('inf')
                            best_pos = (center_x, center_y)
                            
                            for region in regions:
                                # 计算到区域边界的最短距离投影点
                                x_proj = max(region[0] + half_width, 
                                            min(center_x, region[2] - half_width))
                                y_proj = max(region[1] + half_height, 
                                            min(center_y, region[3] - half_height))
                                
                                # 计算到投影点的距离
                                dist = np.sqrt((center_x - x_proj)**2 + (center_y - y_proj)**2)
                                
                                if dist < min_dist:
                                    min_dist = dist
                                    best_pos = (x_proj, y_proj)
                            
                            center_x, center_y = best_pos
            
            constrained_positions[i] = (center_x, center_y)
        
        return constrained_positions
    
    def optimize_placement(self):
        """
        执行GiFt优化过程
        
        返回:
        optimized_positions: 优化后的位置(中心点坐标)
        """
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
        """
        获取DREAMPlace格式的节点位置数组
        
        返回:
        pos: 左下角坐标格式的位置数组，形状为(num_nodes * 2,)
        """
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
        """可视化布局结果"""
        if self.optimized_positions is None:
            self.optimize_placement()
        
        placedb = self.placedb
        positions = self.optimized_positions
        
        plt.figure(figsize=(12, 10))
        
        # 绘制芯片边界
        plt.plot([placedb.xl, placedb.xh, placedb.xh, placedb.xl, placedb.xl], 
                 [placedb.yl, placedb.yl, placedb.yh, placedb.yh, placedb.yl], 
                 'k-', linewidth=2, label='chip boundary')
        
        # 绘制资源区域
        if self.enable_resource_constraints:
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
        for i in range(placedb.num_physical_nodes):
            x, y = positions[i]
            
            if i >= placedb.num_movable_nodes:
                io_x.append(x)
                io_y.append(y)
            elif placedb.lut_mask[i]:
                lut_x.append(x)
                lut_y.append(y)
            elif placedb.flop_mask[i]:
                ff_x.append(x)
                ff_y.append(y)
            elif placedb.dsp_mask[i]:
                dsp_x.append(x)
                dsp_y.append(y)
            elif placedb.ram_mask[i]:
                ram_x.append(x)
                ram_y.append(y)
        
        # 绘制各类节点
        if lut_x: plt.scatter(lut_x, lut_y, c='blue', marker='o', s=10, alpha=0.7, label='LUT')
        if ff_x: plt.scatter(ff_x, ff_y, c='green', marker='s', s=10, alpha=0.7, label='FF')
        if dsp_x: plt.scatter(dsp_x, dsp_y, c='purple', marker='^', s=20, alpha=0.8, label='DSP')
        if ram_x: plt.scatter(ram_x, ram_y, c='orange', marker='d', s=20, alpha=0.8, label='RAM')
        if io_x: plt.scatter(io_x, io_y, c='red', marker='x', s=30, label='IO')
        
        # 标题包含约束信息
        constraints_info = []
        if self.enable_boundary_constraints:
            constraints_info.append("boundary constraint")
        if self.enable_resource_constraints:
            constraints_info.append("resource constraint")
            
        constraints_str = ", ".join(constraints_info) if constraints_info else "no constraint"
        plt.title(f"GiFt Placement Result ({constraints_str})")
        
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
        
        plt.show()