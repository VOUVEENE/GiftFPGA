# gift_init_placer.py - 极速优化
import numpy as np
import scipy.sparse as sp
from scipy.sparse import csc_matrix, diags, identity
import torch
import time
import logging
from collections import defaultdict
from placement_utils import PlacementUtils
import matplotlib.pyplot as plt
from log_config import setup_logging

setup_logging()
logger = logging.getLogger(__name__)

# ====== 通用计时工具 ======
def timed_step(name, func, *args, **kwargs):
    """通用计时器：执行func并记录耗时"""
    start = time.time()
    result = func(*args, **kwargs)
    elapsed = time.time() - start
    logger.info(f"[计时] {name} 耗时: {elapsed:.3f}s")
    return result

def tic():
    return time.time()

def toc(t0, name):
    logger.info(f"[计时] {name} 耗时: {time.time() - t0:.3f}s")

# ====== 尝试导入C++扩展 ======
try:
    import gift_adj_cpp
    CPP_AVAILABLE = True
    logger.info("C++邻接矩阵构建模块已加载")
except ImportError:
    CPP_AVAILABLE = False
    logger.warning("C++扩展未找到，将使用Python实现")

class GiFtGPUFilter:
    """
    极速GPU滤波器 - 全程缓存 + 零拷贝优化
    """
    def __init__(self, adj_mat, device):
        self.adj_mat = adj_mat
        self.device = device
        
        # 全局缓存策略
        self._torch_matrices = {}  # 直接缓存最终的torch稀疏矩阵
        self._gpu_buffers = {}     # GPU内存池
        self._initialized = False
        
        # 预分配标志
        self._buffer_size = None

    def initialize_once(self, position_shape):
        """一次性初始化所有GPU资源"""
        if self._initialized and self._buffer_size == position_shape:
            return
            
        t0 = tic()
        
        # 预分配GPU缓冲区
        self._buffer_size = position_shape
        for i in range(4):  # 预分配4个工作缓冲区
            self._gpu_buffers[f'work_{i}'] = torch.zeros(
                position_shape, device=self.device, dtype=torch.float32
            )
        
        # 批量构建所有σ值的torch稀疏矩阵
        self._build_all_torch_matrices()
        self._initialized = True
        
        toc(t0, "一次性GPU资源初始化")

    def _build_all_torch_matrices(self):
        """批量构建所有需要的torch稀疏矩阵"""
        sigma_values = [4, 2]  # GiFt算法需要的所有σ值
        
        # 预计算基础信息
        adj_csc = csc_matrix(self.adj_mat)
        base_degree = np.array(adj_csc.sum(axis=1)).flatten()
        identity_mat = identity(adj_csc.shape[0])
        
        for sigma in sigma_values:
            # 归一化计算
            rowsum = base_degree + sigma
            d_inv = np.power(rowsum, -0.5)
            d_inv[np.isinf(d_inv)] = 0.
            
            d_mat = diags(d_inv)
            adj_with_loops = adj_csc + sigma * identity_mat
            norm_adj = d_mat.dot(adj_with_loops).dot(d_mat)
            
            # 直接构建torch稀疏张量
            coo_mat = norm_adj.tocoo()
            edge_index = torch.from_numpy(
                np.vstack((coo_mat.row, coo_mat.col))
            ).long().to(self.device)
            edge_weight = torch.from_numpy(coo_mat.data).float().to(self.device)
            
            self._torch_matrices[sigma] = torch.sparse_coo_tensor(
                edge_index, edge_weight, coo_mat.shape,
                device=self.device, dtype=torch.float32
            ).coalesce()

    def get_cell_position(self, k, cell_pos, sigma):
        """保持接口兼容的快速滤波"""
        sparse_mat = self._torch_matrices[sigma]
        
        # 使用预分配的缓冲区
        current = self._gpu_buffers['work_0']
        next_buf = self._gpu_buffers['work_1']
        current.copy_(cell_pos)
        
        mm_t0 = tic()
        for i in range(k):
            step_t = tic()
            if i % 2 == 0:
                next_buf.copy_(torch.sparse.mm(sparse_mat, current))
            else:
                current.copy_(torch.sparse.mm(sparse_mat, next_buf))
            toc(step_t, f"GPUFilter 稀疏乘法（第{i+1}步）")
            
        result = next_buf if k % 2 == 1 else current
        toc(mm_t0, f"GPUFilter 稀疏乘法（累计 {k} 次）")
        return result.clone()

    def fast_filter(self, initial_pos_tensor, filter_configs):
        """
        超快速滤波 - 零拷贝 + 流水线
        filter_configs: [(sigma, k), ...] 如 [(4,4), (4,2), (2,2)]
        """
        results = []
        work_buffers = [self._gpu_buffers[f'work_{i}'] for i in range(4)]
        
        for i, (sigma, k) in enumerate(filter_configs):
            sparse_mat = self._torch_matrices[sigma]
            
            # 使用预分配的缓冲区，避免内存分配
            current = work_buffers[i % 4]
            next_buf = work_buffers[(i + 1) % 4]
            
            # 复制输入
            current.copy_(initial_pos_tensor)
            
            # 执行k次矩阵乘法，缓冲区乒乓
            for step in range(k):
                if step % 2 == 0:
                    next_buf.copy_(torch.sparse.mm(sparse_mat, current))
                else:
                    current.copy_(torch.sparse.mm(sparse_mat, next_buf))
            
            # 保存结果（复制最终结果到新缓冲区）
            result_buf = work_buffers[(i + 2) % 4]
            if k % 2 == 1:
                result_buf.copy_(next_buf)
            else:
                result_buf.copy_(current)
            results.append(result_buf)
        
        return results

class GiFtFPGAPlacer:
    """
    基于图信号处理的FPGA布局加速器 - 极速优化版
    """
    def __init__(self, placedb, params):
        self.placedb = placedb
        self.params = params
        self.scale = getattr(params, 'gift_scale', 0.7)
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        
        # 约束开关 - 默认优化配置
        self.enable_boundary_constraints = getattr(params, 'gift_enable_boundary_constraints', True)
        self.enable_resource_constraints = getattr(params, 'gift_enable_resource_constraints', False)  # 默认关闭
        self.enable_hpwl = getattr(params, 'gift_enable_hpwl', False)  # 默认关闭
        self.hpwl_backend = getattr(params, 'gift_hpwl_backend', 'numpy').lower()
        
        # 布局边界
        self.xl, self.yl = placedb.xl, placedb.yl
        self.xh, self.yh = placedb.xh, placedb.yh
        
        # 缓存
        self.adjacency_matrix = None
        self.gpu_filter = None
        self.initial_positions = None
        self.optimized_positions = None
        
        # 资源区域（仅在启用时导入）
        self.resource_regions = {}
        if self.enable_resource_constraints:
            timed_step("导入资源区域", self.import_resource_regions)
        
        logger.info(f"GiFt优化版本初始化完成，使用设备: {self.device}, C++: {'启用' if CPP_AVAILABLE else '禁用'}, HPWL(enable={self.enable_hpwl}, backend={self.hpwl_backend})")

    def import_resource_regions(self):
        """快速导入资源区域"""
        placedb = self.placedb
        resource_types = ['LUT', 'FF', 'DSP', 'RAM']
        if hasattr(placedb, 'region_boxes') and placedb.region_boxes:
            for i, regions in enumerate(placedb.region_boxes):
                if i < len(resource_types):
                    self.resource_regions[resource_types[i]] = regions
                    logger.info(f"导入资源区域 - {resource_types[i]}: {len(regions)}个区域")

    def build_adjacency_matrix(self):
        """快速构建邻接矩阵"""
        if self.adjacency_matrix is not None:
            return
            
        if CPP_AVAILABLE:
            return timed_step("C++邻接矩阵构建", self.build_adjacency_matrix_cpp)
        else:
            return timed_step("Python邻接矩阵构建", self._build_adj_python_fast)

    def build_adjacency_matrix_cpp(self):
        placedb = self.placedb
        start_time = tic()

        flat_netpin = placedb.flat_net2pin_map.astype(np.int32)
        netpin_start = placedb.flat_net2pin_start_map.astype(np.int32)
        pin2node_map = placedb.pin2node_map.astype(np.int32)
        net_weights = np.ones(placedb.num_nets, dtype=np.float32)
        net_mask = np.ones(placedb.num_nets, dtype=np.int32)

        data, rows, cols = gift_adj_cpp.adj_matrix_forward(
            flat_netpin, netpin_start, pin2node_map,
            net_weights, net_mask, placedb.num_physical_nodes
        )

        self.adjacency_matrix = sp.coo_matrix(
            (data, (rows, cols)),
            shape=(placedb.num_physical_nodes, placedb.num_physical_nodes),
            dtype=np.float32
        ).tocsr()

        build_time = time.time() - start_time
        density = self.adjacency_matrix.nnz / (placedb.num_physical_nodes ** 2) * 100
        logger.info(f"C++邻接矩阵构建完成，耗时: {build_time:.3f}s, 非零元素: {self.adjacency_matrix.nnz:,}, 密度: {density:.4f}%")

    def _build_adj_python_fast(self):
        """Python快速邻接矩阵构建"""
        placedb = self.placedb
        n = placedb.num_physical_nodes
        
        edges = []
        weights = []
        
        for net_id in range(placedb.num_nets):
            pin_start = placedb.flat_net2pin_start_map[net_id]
            pin_end = placedb.flat_net2pin_start_map[net_id + 1]
            num_pins = pin_end - pin_start
            if num_pins < 2:
                continue

            weight = 2.0 / num_pins
            nodes = []
            for j in range(pin_start, pin_end):
                flat_pin_id = placedb.flat_net2pin_map[j]
                node_id = placedb.pin2node_map[flat_pin_id]
                if node_id < n:
                    nodes.append(node_id)

            for i in range(len(nodes)):
                for j in range(i + 1, len(nodes)):
                    edges.extend([(nodes[i], nodes[j]), (nodes[j], nodes[i])])
                    weights.extend([weight, weight])

        if not edges:
            self.adjacency_matrix = sp.csr_matrix((n, n), dtype=np.float32)
            return

        rows, cols = zip(*edges)
        self.adjacency_matrix = sp.coo_matrix((weights, (rows, cols)), shape=(n, n)).tocsr()

    def set_preset_center(self, center_x, center_y):
        """设置预设中心"""
        self._preset_center = (center_x, center_y)
        logger.info(f"使用预设中心位置: ({center_x:.2f}, {center_y:.2f})")

    def initialize_positions(self):
        """快速初始化位置"""
        if self.initial_positions is not None:
            return self.initial_positions
            
        t0 = tic()
        placedb = self.placedb
        n = placedb.num_physical_nodes

        # 使用预设中心或计算中心
        if hasattr(self, '_preset_center'):
            center_x, center_y = self._preset_center
        else:
            center_x, center_y = PlacementUtils.calculate_initial_center(placedb)

        # 快速生成初始位置
        positions = np.zeros((n, 2), dtype=np.float32)
        
        # 固定节点
        for i in range(placedb.num_movable_nodes, n):
            positions[i, 0] = placedb.node_x[i] + placedb.node_size_x[i] / 2
            positions[i, 1] = placedb.node_y[i] + placedb.node_size_y[i] / 2

        # 可移动节点生成
        fixed_positions = positions[placedb.num_movable_nodes:]
        random_initial = timed_step(
            "生成随机初始位置(generate_initial_locations)",
            self.generate_initial_locations, fixed_positions, placedb.num_movable_nodes, self.scale
        )
        positions[:placedb.num_movable_nodes] = random_initial

        self.initial_positions = positions
        toc(t0, "初始化位置(initialize_positions)")
        return positions

    def generate_initial_locations(self, fixed_cell_location, movable_num, scale):
        if len(fixed_cell_location) == 0:
            if hasattr(self, '_preset_center'):
                xcenter, ycenter = self._preset_center
            else:
                xcenter = (self.xh + self.xl) / 2
                ycenter = (self.yh + self.yl) / 2
            x_range = self.xh - self.xl
            y_range = self.yh - self.yl
        else:
            x_min, x_max = np.min(fixed_cell_location[:, 0]), np.max(fixed_cell_location[:, 0])
            y_min, y_max = np.min(fixed_cell_location[:, 1]), np.max(fixed_cell_location[:, 1])
            x_range = x_max - x_min
            y_range = y_max - y_min
            
            if hasattr(self, '_preset_center'):
                xcenter, ycenter = self._preset_center
            else:
                xcenter = (x_max + x_min) / 2
                ycenter = (y_max + y_min) / 2

        # 向量化生成随机分布
        random_pos = np.random.rand(int(movable_num), 2)
        random_pos[:, 0] = ((random_pos[:, 0] - 0.5) * x_range * scale) + xcenter
        random_pos[:, 1] = ((random_pos[:, 1] - 0.5) * y_range * scale) + ycenter
        return random_pos

    def apply_gift_filters(self, initial_positions):
        """极速GiFt滤波"""
        logger.info("应用GiFt滤波器...")
        if self.adjacency_matrix is None:
            self.build_adjacency_matrix()

        t0 = tic()
        
        # 初始化GPU滤波器（仅一次）
        if self.gpu_filter is None:
            self.gpu_filter = GiFtGPUFilter(self.adjacency_matrix, self.device)
        
        # 初始化GPU资源
        self.gpu_filter.initialize_once(initial_positions.shape)

        # 零拷贝数据上传
        t_to_torch = tic()
        gpu_pos = self.gpu_filter._gpu_buffers['work_0']
        gpu_pos.copy_(torch.from_numpy(initial_positions))
        toc(t_to_torch, "初始位置 转换为 Torch 张量")

        # 使用快速批量滤波
        filter_results = self.gpu_filter.fast_filter(
            gpu_pos, [(4, 4), (4, 2), (2, 2)]
        )

        # 快速组合结果
        t_combine = tic()
        location = (0.2 * filter_results[0] + 
                   0.7 * filter_results[1] + 
                   0.1 * filter_results[2])
        toc(t_combine, "滤波结果组合")

        # 快速回传
        t_back = tic()
        if self.device.type == 'cuda':
            torch.cuda.synchronize()
        optimized_positions = location.cpu().numpy()
        toc(t_back, "Torch 张量 转回 Numpy")
        
        # 保持固定节点位置
        num_movable = self.placedb.num_movable_nodes
        optimized_positions[num_movable:] = initial_positions[num_movable:]

        toc(t0, "应用GiFt滤波器(apply_gift_filters)")
        return optimized_positions

    def apply_placement_constraints(self, positions):
        """快速约束应用"""
        t0 = tic()
        
        if not self.enable_boundary_constraints and not self.enable_resource_constraints:
            toc(t0, "apply_placement_constraints (跳过)")
            return positions
        
        constrained_positions = positions.copy()
        placedb = self.placedb
        num_movable = placedb.num_movable_nodes
        
        # 边界约束 - 向量化处理
        if self.enable_boundary_constraints:
            boundary_t0 = tic()
            
            half_widths = placedb.node_size_x[:num_movable] / 2
            half_heights = placedb.node_size_y[:num_movable] / 2
            
            x_min_bounds = placedb.xl + half_widths
            x_max_bounds = placedb.xh - half_widths
            y_min_bounds = placedb.yl + half_heights
            y_max_bounds = placedb.yh - half_heights
            
            constrained_positions[:num_movable, 0] = np.clip(
                positions[:num_movable, 0], x_min_bounds, x_max_bounds
            )
            constrained_positions[:num_movable, 1] = np.clip(
                positions[:num_movable, 1], y_min_bounds, y_max_bounds
            )
            
            toc(boundary_t0, "边界约束(向量化)")
        
        # 资源约束 - 如果启用
        if self.enable_resource_constraints and self.resource_regions:
            resource_t0 = tic()
            for i in range(num_movable):
                center_x = constrained_positions[i, 0]
                center_y = constrained_positions[i, 1]
                half_width = placedb.node_size_x[i] / 2
                half_height = placedb.node_size_y[i] / 2
                
                resource_type = self._get_node_resource_type(i)
                if resource_type and resource_type in self.resource_regions:
                    center_x, center_y = self._apply_resource_constraints(
                        center_x, center_y, half_width, half_height, resource_type
                    )
                    constrained_positions[i] = (center_x, center_y)
            toc(resource_t0, "资源约束(逐节点)")
        
        toc(t0, "apply_placement_constraints")
        return constrained_positions

    def _get_node_resource_type(self, node_id):
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
        regions = self.resource_regions[resource_type]
        for region in regions:
            if (region[0] + half_width <= center_x <= region[2] - half_width and
                region[1] + half_height <= center_y <= region[3] - half_height):
                return center_x, center_y

        min_dist = float('inf')
        best_pos = (center_x, center_y)
        for region in regions:
            x_proj = max(region[0] + half_width, min(center_x, region[2] - half_width))
            y_proj = max(region[1] + half_height, min(center_y, region[3] - half_height))
            dist = np.sqrt((center_x - x_proj)**2 + (center_y - y_proj)**2)
            if dist < min_dist:
                min_dist = dist
                best_pos = (x_proj, y_proj)
        return best_pos

    def optimize_placement(self):
        """主优化流程 - 极速版"""
        total_start = tic()

        if self.initial_positions is None:
            self.initial_positions = timed_step("初始化位置(initialize_positions)", self.initialize_positions)

        self.optimized_positions = timed_step("应用GiFt滤波器(apply_gift_filters)",
                                             self.apply_gift_filters, self.initial_positions)

        self.optimized_positions = timed_step("应用布局约束(apply_placement_constraints)",
                                             self.apply_placement_constraints, self.optimized_positions)

        if self.enable_hpwl:
            if self.hpwl_backend == 'numpy':
                initial_wl = timed_step("初始HPWL计算(calculate_hpwl_numpy)", self.calculate_hpwl_numpy, self.initial_positions)
                optimized_wl = timed_step("优化后HPWL计算(calculate_hpwl_numpy)", self.calculate_hpwl_numpy, self.optimized_positions)
            else:
                initial_wl = timed_step("初始HPWL计算(calculate_hpwl)", self.calculate_hpwl, self.initial_positions)
                optimized_wl = timed_step("优化后HPWL计算(calculate_hpwl)", self.calculate_hpwl, self.optimized_positions)

            total_time = time.time() - total_start
            logger.info(f"GiFt优化完成，总耗时: {total_time:.3f}s")
            logger.info(f"初始HPWL: {initial_wl:.2f}，优化后HPWL: {optimized_wl:.2f}")
            if optimized_wl < initial_wl:
                improvement = (initial_wl - optimized_wl) / initial_wl * 100
                logger.info(f"GiFt优化有效，HPWL减少了 {improvement:.2f}%")
            else:
                degradation = (optimized_wl - initial_wl) / initial_wl * 100
                logger.warning(f"GiFt优化后HPWL增加了 {degradation:.2f}%")
        else:
            total_time = time.time() - total_start
            logger.info(f"GiFt优化完成（跳过HPWL），总耗时: {total_time:.3f}s")

        return self.optimized_positions

    def get_dreamplace_positions(self):
        """输出给DREAMPlace的位置格式"""
        if self.optimized_positions is None:
            self.optimize_placement()

        placedb = self.placedb
        pos = np.zeros(placedb.num_nodes * 2, dtype=placedb.dtype)

        pos[0:placedb.num_physical_nodes] = placedb.node_x
        pos[placedb.num_nodes:placedb.num_nodes+placedb.num_physical_nodes] = placedb.node_y

        for i in range(placedb.num_movable_nodes):
            center_x = self.optimized_positions[i, 0]
            center_y = self.optimized_positions[i, 1]
            pos[i] = center_x - placedb.node_size_x[i] / 2
            pos[placedb.num_nodes + i] = center_y - placedb.node_size_y[i] / 2

        return pos

    def calculate_hpwl(self, positions):
        """原版HPWL计算"""
        t0 = tic()
        placedb = self.placedb
        total_wl = 0

        for net_id in range(placedb.num_nets):
            pin_start = placedb.flat_net2pin_start_map[net_id]
            pin_end = placedb.flat_net2pin_start_map[net_id + 1]
            if pin_end - pin_start < 2:
                continue

            x_min = float('inf')
            x_max = float('-inf')
            y_min = float('inf')
            y_max = float('-inf')

            for pin_id in range(pin_start, pin_end):
                flat_pin_id = placedb.flat_net2pin_map[pin_id]
                node_id = placedb.pin2node_map[flat_pin_id]
                if node_id < placedb.num_physical_nodes:
                    node_center_x = positions[node_id, 0]
                    node_center_y = positions[node_id, 1]
                    pin_offset_x = placedb.pin_offset_x[flat_pin_id]
                    pin_offset_y = placedb.pin_offset_y[flat_pin_id]

                    pin_x = node_center_x + pin_offset_x - placedb.node_size_x[node_id] / 2
                    pin_y = node_center_y + pin_offset_y - placedb.node_size_y[node_id] / 2

                    x_min = min(x_min, pin_x)
                    x_max = max(x_max, pin_x)
                    y_min = min(y_min, pin_y)
                    y_max = max(y_max, pin_y)

            if x_max > x_min and y_max > y_min:
                net_wl = (x_max - x_min) + (y_max - y_min)
                total_wl += net_wl

        toc(t0, "calculate_hpwl")
        return total_wl

    def calculate_hpwl_numpy(self, positions):
        """快速NumPy HPWL计算"""
        t0 = tic()
        placedb = self.placedb

        flat_pin = placedb.flat_net2pin_map
        pin2node = placedb.pin2node_map
        node_id = pin2node[flat_pin]
        cx = positions[node_id, 0]
        cy = positions[node_id, 1]
        sx = placedb.node_size_x[node_id]
        sy = placedb.node_size_y[node_id]
        ox = placedb.pin_offset_x[flat_pin]
        oy = placedb.pin_offset_y[flat_pin]

        px = cx + ox - 0.5 * sx
        py = cy + oy - 0.5 * sy

        starts = placedb.flat_net2pin_start_map
        counts = starts[1:] - starts[:-1]
        valid = counts >= 2
        if not np.any(valid):
            toc(t0, "calculate_hpwl_numpy")
            return 0.0

        s = starts[:-1][valid]
        xmin = np.minimum.reduceat(px, s)
        xmax = np.maximum.reduceat(px, s)
        ymin = np.minimum.reduceat(py, s)
        ymax = np.maximum.reduceat(py, s)

        rx = xmax - xmin
        ry = ymax - ymin
        positive = (rx > 0) & (ry > 0)
        hpwl = (rx + ry) * positive

        total = float(hpwl.sum())
        toc(t0, "calculate_hpwl_numpy")
        return total

    def visualize_placement(self, output_file=None, show_nets=False):
        """可视化布局结果"""
        if self.optimized_positions is None:
            logger.warning("没有优化位置数据，无法进行可视化")
            return
        return timed_step("可视化(optimized)", self.visualize_placement_with_positions,
                          self.optimized_positions, output_file, "GiFt Placement Result", show_nets)

    def visualize_placement_with_positions(self, positions, output_file=None, title="GiFt Placement", show_nets=False):
        """使用给定位置进行可视化"""
        t0 = tic()
        if positions is None:
            logger.warning("位置数组为空，无法进行可视化")
            return

        placedb = self.placedb
        plt.figure(figsize=(12, 10))

        # 芯片边界
        plt.plot([placedb.xl, placedb.xh, placedb.xh, placedb.xl, placedb.xl],
                 [placedb.yl, placedb.yl, placedb.yh, placedb.yh, placedb.yl],
                 'k-', linewidth=2, label='chip boundary')

        # 资源区域
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

        # 分类节点
        lut_x, lut_y = [], []
        ff_x, ff_y = [], []
        dsp_x, dsp_y = [], []
        ram_x, ram_y = [], []
        io_x, io_y = [], []

        for i in range(min(placedb.num_physical_nodes, len(positions))):
            x, y = positions[i]
            if i >= placedb.num_movable_nodes:
                io_x.append(x); io_y.append(y)
            elif hasattr(placedb, 'lut_mask') and placedb.lut_mask[i]:
                lut_x.append(x); lut_y.append(y)
            elif hasattr(placedb, 'flop_mask') and placedb.flop_mask[i]:
                ff_x.append(x); ff_y.append(y)
            elif hasattr(placedb, 'dsp_mask') and placedb.dsp_mask[i]:
                dsp_x.append(x); dsp_y.append(y)
            elif hasattr(placedb, 'ram_mask') and placedb.ram_mask[i]:
                ram_x.append(x); ram_y.append(y)

        if lut_x: plt.scatter(lut_x, lut_y, c='blue', marker='o', s=10, alpha=0.7, label='LUT')
        if ff_x: plt.scatter(ff_x, ff_y, c='green', marker='s', s=10, alpha=0.7, label='FF')
        if dsp_x: plt.scatter(dsp_x, dsp_y, c='purple', marker='^', s=20, alpha=0.8, label='DSP')
        if ram_x: plt.scatter(ram_x, ram_y, c='orange', marker='d', s=20, alpha=0.8, label='RAM')
        if io_x:  plt.scatter(io_x,  io_y,  c='red', marker='x', s=30, label='IO')

        plt.title(title)
        plt.legend()
        plt.grid(True, alpha=0.3)
        plt.xlabel('X')
        plt.ylabel('Y')

        if output_file:
            plt.savefig(output_file, dpi=300, bbox_inches='tight')
            logger.info(f"可视化结果已保存到 {output_file}")
        else:
            plt.show()

        plt.close()
        toc(t0, "visualize_placement_with_positions")