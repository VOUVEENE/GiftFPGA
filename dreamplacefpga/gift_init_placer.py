# gift_placer.py
import numpy as np
import scipy.sparse as sp
import matplotlib.pyplot as plt
from collections import defaultdict
import time
import os
import logging

class GiFtFPGAPlacer:
    """
    基于图信号处理的FPGA布局加速器
    基于论文: The Power of Graph Signal Processing for Chip Placement Acceleration
    
    集成到DREAMPlaceFPGA的版本
    
    特性:
    - 区域约束和资源约束分开处理
    - 可通过参数控制是否使用约束
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
        self.alpha0 = getattr(params, 'gift_alpha0', 0.1)
        self.alpha1 = getattr(params, 'gift_alpha1', 0.7)
        self.alpha2 = getattr(params, 'gift_alpha2', 0.2)
        
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
        
        # 资源区域定义
        self.resource_regions = {}
        # 从placedb导入资源区域信息
        self.import_resource_regions()
        
        logging.info(f"GiFt初始化完成 - 边界约束: {'启用' if self.enable_boundary_constraints else '禁用'}, 资源约束: {'启用' if self.enable_resource_constraints else '禁用'}")
    
    def import_resource_regions(self):
        """
        从placedb导入资源区域信息
        """
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
        """
        计算初始中心位置，与BasicPlace方法相同
        
        返回:
        (initLocX, initLocY): 初始中心位置坐标
        """
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
        """
        构建节点之间的邻接矩阵
        
        返回:
        A: 邻接矩阵（稀疏矩阵）
        """
        logging.info("构建邻接矩阵...")
        placedb = self.placedb
        n = placedb.num_physical_nodes
        A = sp.lil_matrix((n, n), dtype=np.float32)
        
        # 使用clique模型构建邻接矩阵
        for net_id in range(placedb.num_nets):
            # 获取该网络的引脚
            pin_start = placedb.flat_net2pin_start_map[net_id]
            pin_end = placedb.flat_net2pin_start_map[net_id + 1]
            
            if pin_end - pin_start < 2:
                continue  # 跳过只有一个引脚的网络
            
            # 收集此网络连接的所有节点
            nodes = set()
            for pin_id in range(pin_start, pin_end):
                flat_pin_id = placedb.flat_net2pin_map[pin_id]
                node_id = placedb.pin2node_map[flat_pin_id]
                nodes.add(node_id)
            
            nodes = list(nodes)
            if len(nodes) < 2:
                continue
            
            # 对于该网络中的每对节点，添加边
            weight = 2.0 / len(nodes)  # 使用 2/M 作为权重，如论文中所述
            for i in range(len(nodes)):
                for j in range(i+1, len(nodes)):
                    A[nodes[i], nodes[j]] += weight
                    A[nodes[j], nodes[i]] += weight
        
        return A.tocsr()  # 转换为CSR格式以加速计算
    
    def get_normalized_adjacency(self, A, sigma=2):
        """
        计算归一化邻接矩阵(增加自环)
        
        参数:
        A: 邻接矩阵
        sigma: 自环权重
        
        返回:
        A_tilde: 归一化的邻接矩阵(增加自环)
        """
        # 增加自环
        n = A.shape[0]
        I = sp.eye(n)
        A_sigma = A + sigma * I
        
        # 计算度矩阵
        row_sum = np.array(A_sigma.sum(axis=1)).flatten()
        
        # 避免除以零（孤立节点）
        row_sum[row_sum == 0] = 1.0
        
        # 计算D^(-1/2) * A * D^(-1/2)
        D_sqrt_inv = sp.diags(1.0 / np.sqrt(row_sum))
        A_tilde = D_sqrt_inv @ A_sigma @ D_sqrt_inv
        
        return A_tilde
    
    def initialize_positions(self):
        """
        初始化节点位置
        
        返回:
        initial_positions: 初始位置，形状为(n, 2)的numpy数组
        """
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
        应用GiFt算法计算优化的位置
        
        参数:
        initial_positions: 初始位置，形状为(n, 2)的numpy数组
        
        返回:
        optimized_positions: 优化后的位置，形状为(n, 2)的numpy数组
        """
        logging.info(f"应用GiFt滤波器... (alpha0={self.alpha0}, alpha1={self.alpha1}, alpha2={self.alpha2})")
        
        # 构建邻接矩阵
        A = self.build_adjacency_matrix()
        
        # 计算不同的滤波器
        A_tilde_2 = self.get_normalized_adjacency(A, sigma=2)
        A_tilde_4 = self.get_normalized_adjacency(A, sigma=4)
        
        # 计算不同的滤波器组合
        A_tilde_2_2 = A_tilde_2 @ A_tilde_2  # 高通滤波器
        A_tilde_2_4 = A_tilde_4 @ A_tilde_4  # 中通滤波器
        A_tilde_4_4 = A_tilde_4 @ A_tilde_4 @ A_tilde_4 @ A_tilde_4  # 低通滤波器
        
        # 按照论文中的公式：g' = α₀Ã²₂g + α₁Ã²₄g + α₂Ã⁴₄g
        g_prime_x = (
            self.alpha0 * A_tilde_2_2.dot(initial_positions[:, 0]) +
            self.alpha1 * A_tilde_2_4.dot(initial_positions[:, 0]) +
            self.alpha2 * A_tilde_4_4.dot(initial_positions[:, 0])
        )
        
        g_prime_y = (
            self.alpha0 * A_tilde_2_2.dot(initial_positions[:, 1]) +
            self.alpha1 * A_tilde_2_4.dot(initial_positions[:, 1]) +
            self.alpha2 * A_tilde_4_4.dot(initial_positions[:, 1])
        )
        
        optimized_positions = np.column_stack((g_prime_x, g_prime_y))
        return optimized_positions
    
    def apply_placement_constraints(self, positions):
        """
        应用布局范围约束，确保所有节点都在芯片边界内
        并满足资源区域约束
        
        参数:
        positions: 优化后的位置
        
        返回:
        constrained_positions: 满足约束的位置
        """
        logging.info(f"应用布局约束... (边界约束: {'启用' if self.enable_boundary_constraints else '禁用'}, 资源约束: {'启用' if self.enable_resource_constraints else '禁用'})")
        placedb = self.placedb
        constrained_positions = positions.copy()
        
        # 保持固定节点不变
        for i in range(placedb.num_movable_nodes, placedb.num_physical_nodes):
            constrained_positions[i] = positions[i]
        
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
        # 初始化位置
        if self.initial_positions is None:
            self.initialize_positions()
        
        # 应用GiFt算法
        self.optimized_positions = self.apply_gift(self.initial_positions)
        
        # 应用布局约束
        self.optimized_positions = self.apply_placement_constraints(self.optimized_positions)
        
        # 还原固定节点的位置（以防万一）
        for i in range(self.placedb.num_movable_nodes, self.placedb.num_physical_nodes):
            self.optimized_positions[i] = self.initial_positions[i]
        
        # 计算初始和优化后的HPWL
        initial_wl = self.calculate_hpwl(self.initial_positions)
        optimized_wl = self.calculate_hpwl(self.optimized_positions)
        
        # logging.info(f"GiFt初始化完成，初始HPWL: {initial_wl:.2f}，优化后HPWL: {optimized_wl:.2f}")
        # if optimized_wl < initial_wl:
        #     logging.info(f"GiFt优化有效，HPWL减少了 {(initial_wl - optimized_wl) / initial_wl * 100:.2f}%")
        # else:
        #     logging.warning("GiFt优化未能减少HPWL，可能需要调整参数")
        
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
        """
        可视化布局结果
        
        参数:
        output_file: 输出图像文件名
        show_nets: 是否显示网络连接
        """
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
            for resource_type, regions in self.resource_regions.items():
                for j, region in enumerate(regions):
                    x_min, y_min, x_max, y_max = region
                    label = f'{resource_type} region' if j == 0 else None  # 只为第一个区域添加标签
                    plt.plot([x_min, x_max, x_max, x_min, x_min], 
                             [y_min, y_min, y_max, y_max, y_min], 
                             '--', linewidth=1, label=label)
        
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
        plt.scatter(lut_x, lut_y, c='blue', marker='o', s=10, alpha=0.7, label='LUT')
        plt.scatter(ff_x, ff_y, c='green', marker='s', s=10, alpha=0.7, label='FF')
        plt.scatter(dsp_x, dsp_y, c='purple', marker='^', s=20, alpha=0.8, label='DSP')
        plt.scatter(ram_x, ram_y, c='orange', marker='d', s=20, alpha=0.8, label='RAM')
        plt.scatter(io_x, io_y, c='red', marker='x', s=30, label='IO')
        
        # 绘制网络连接（如果需要）
        if show_nets and placedb.num_nets < 1000:  # 如果网络太多，不绘制避免混乱
            for net_id in range(placedb.num_nets):
                # 获取网络中的节点
                pin_start = placedb.flat_net2pin_start_map[net_id]
                pin_end = placedb.flat_net2pin_start_map[net_id + 1]
                
                if pin_end - pin_start < 2:
                    continue
                
                # 收集网络中的节点位置
                net_nodes = set()
                for pin_id in range(pin_start, pin_end):
                    flat_pin_id = placedb.flat_net2pin_map[pin_id]
                    node_id = placedb.pin2node_map[flat_pin_id]
                    if node_id < placedb.num_physical_nodes:
                        net_nodes.add(node_id)
                
                net_nodes = list(net_nodes)
                if len(net_nodes) < 2:
                    continue
                
                # 绘制连接线（浅色）
                for i in range(len(net_nodes)):
                    for j in range(i+1, len(net_nodes)):
                        plt.plot([positions[net_nodes[i], 0], positions[net_nodes[j], 0]], 
                                 [positions[net_nodes[i], 1], positions[net_nodes[j], 1]], 
                                 'g-', alpha=0.1, linewidth=0.5)
        
        # 标题包含约束信息
        constraints_info = []
        if self.enable_boundary_constraints:
            constraints_info.append("boundary constraint")
        if self.enable_resource_constraints:
            constraints_info.append("resource constraint")
            
        constraints_str = "、".join(constraints_info) if constraints_info else "no constraint"
        plt.title(f"GiFt Placement Result ({constraints_str})")
        
        # 移除图例中的重复项
        handles, labels = plt.gca().get_legend_handles_labels()
        by_label = dict(zip(labels, handles))
        plt.legend(by_label.values(), by_label.keys())
        
        plt.grid(True, linestyle='--', alpha=0.6)
        plt.xlabel('X')
        plt.ylabel('Y')
        
        # 保存图像
        if output_file:
            plt.savefig(output_file, dpi=300, bbox_inches='tight')
            logging.info(f"可视化结果已保存到 {output_file}")
        
        plt.show()