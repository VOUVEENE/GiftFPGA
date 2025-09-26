# gift_init_placer.py - 保守加速 + HPWL开关版本
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
    优化版GPU滤波器 - 批量处理归一化矩阵
    """
    def __init__(self, adj_mat, device):
        self.adj_mat = adj_mat
        self.device = device
        self.norm_adj = None
        
        # 按 σ 缓存：相同 σ 复用归一化邻接与 Torch 稀疏矩阵
        self._norm_cache = {}
        self._torch_cache = {}
        
        # 预计算基础矩阵信息，避免重复转换
        self._base_adj_csc = None
        self._base_degree = None
        self._identity = None
        self._prepare_base_matrices()

    def _prepare_base_matrices(self):
        """预计算基础矩阵，避免重复转换"""
        t0 = tic()
        self._base_adj_csc = csc_matrix(self.adj_mat)
        self._base_degree = np.array(self._base_adj_csc.sum(axis=1)).flatten()
        self._identity = identity(self._base_adj_csc.shape[0])
        toc(t0, "预计算基础矩阵")

    def batch_train(self, sigma_list):
        """批量计算多个σ值的归一化矩阵"""
        t0 = tic()
        
        new_sigmas = [s for s in sigma_list if s not in self._norm_cache]
        if not new_sigmas:
            logger.info(f"[批量缓存复用] 所有σ值 {sigma_list} 已缓存")
            return
        
        for sigma in new_sigmas:
            # 复用预计算的度信息
            rowsum = self._base_degree + sigma
            d_inv = np.power(rowsum, -0.5)
            d_inv[np.isinf(d_inv)] = 0.
            
            d_mat = diags(d_inv)
            adj_with_self_loops = self._base_adj_csc + sigma * self._identity
            norm_adj = d_mat.dot(adj_with_self_loops).dot(d_mat)
            
            self._norm_cache[sigma] = norm_adj
            
        toc(t0, f"批量归一化矩阵构建 (新增σ={new_sigmas})")

    def _get_torch_sparse(self, sigma):
        """获取或创建torch稀疏张量"""
        if sigma in self._torch_cache:
            return self._torch_cache[sigma]
            
        assert sigma in self._norm_cache, f"σ={sigma} 的归一化矩阵未准备"
        
        build_torch_t = tic()
        trainAdj = self._norm_cache[sigma].tocoo()
        edge_index = np.vstack((trainAdj.row, trainAdj.col)).transpose()
        edge_index = torch.from_numpy(edge_index).long().t().to(self.device)
        edge_weight = torch.from_numpy(trainAdj.data).float().to(self.device)

        norm_adj_torch = torch.sparse.FloatTensor(edge_index, edge_weight).to(self.device)
        self._torch_cache[sigma] = norm_adj_torch
        toc(build_torch_t, f"构建 Torch 稀疏矩阵 (σ={sigma})")
        
        return norm_adj_torch

    def get_cell_position(self, k, cell_pos, sigma):
        """执行k次稀疏矩阵乘法（保持向后兼容）"""
        # 确保归一化矩阵已准备
        if sigma not in self._norm_cache:
            self.batch_train([sigma])
            
        norm_adj_torch = self._get_torch_sparse(sigma)

        mm_total_t = tic()
        for i in range(k):
            step_t = tic()
            cell_pos = torch.sparse.mm(norm_adj_torch, cell_pos)
            toc(step_t, f"GPUFilter 稀疏乘法（第{i+1}步）")
        toc(mm_total_t, f"GPUFilter 稀疏乘法（累计 {k} 次）")

        if cell_pos.is_cuda:
            torch.cuda.empty_cache()

        return cell_pos

    def batch_filter(self, initial_pos, configs):
        """
        批量执行多种滤波配置
        configs: [(sigma, k), ...] 如 [(4, 4), (4, 2), (2, 2)]
        """
        # 预先计算所有需要的归一化矩阵
        sigma_values = list(set(config[0] for config in configs))
        self.batch_train(sigma_values)
        
        results = []
        for sigma, k in configs:
            t0 = tic()
            norm_adj_torch = self._get_torch_sparse(sigma)
            
            # 执行k次矩阵乘法
            cell_pos = initial_pos
            for i in range(k):
                cell_pos = torch.sparse.mm(norm_adj_torch, cell_pos)
                
            results.append(cell_pos)
            toc(t0, f"批量滤波 (σ={sigma}, k={k})")
            
        # 清理GPU缓存
        if initial_pos.is_cuda:
            torch.cuda.empty_cache()
            
        return results


class GiFtFPGAPlacer:
    """
    基于图信号处理的FPGA布局加速器 - 保守加速 + HPWL开关
    """
    def __init__(self, placedb, params):
        self.placedb = placedb
        self.params = params

        # GiFt参数（与DREAMPlace一致）
        self.scale = getattr(params, 'gift_scale', 0.7)

        # 设备
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

        # 约束
        self.enable_boundary_constraints = getattr(params, 'gift_enable_boundary_constraints', True)
        self.enable_resource_constraints = getattr(params, 'gift_enable_resource_constraints', True)

        # HPWL 开关与后端
        self.enable_hpwl = getattr(params, 'gift_enable_hpwl', True)
        # "naive" 完全保留你现有逐网循环实现；"numpy" 启用可选的NumPy向量化
        self.hpwl_backend = getattr(params, 'gift_hpwl_backend', 'naive').lower()

        # 布局边界
        self.xl, self.yl = placedb.xl, placedb.yl
        self.xh, self.yh = placedb.xh, placedb.yh

        # 数据容器
        self.adjacency_matrix = None
        self.initial_positions = None
        self.optimized_positions = None

        # 资源区域
        self.resource_regions = {}
        timed_step("导入资源区域", self.import_resource_regions)

        logger.info(
            f"GiFt优化版本初始化完成，使用设备: {self.device}, "
            f"C++: {'启用' if CPP_AVAILABLE else '禁用'}, "
            f"HPWL(enable={self.enable_hpwl}, backend={self.hpwl_backend})"
        )

    def import_resource_regions(self):
        placedb = self.placedb
        resource_types = ['LUT', 'FF', 'DSP', 'RAM']
        if hasattr(placedb, 'region_boxes') and placedb.region_boxes:
            for i, regions in enumerate(placedb.region_boxes):
                if i < len(resource_types):
                    resource_type = resource_types[i]
                    self.resource_regions[resource_type] = regions
                    logger.info(f"导入资源区域 - {resource_type}: {len(regions)}个区域")

    # ====== 邻接矩阵构建 ======
    def build_adjacency_matrix(self):
        if CPP_AVAILABLE:
            return timed_step("C++邻接矩阵构建", self.build_adjacency_matrix_cpp)
        else:
            return timed_step("Python邻接矩阵构建", self.build_adjacency_matrix_python)

    def build_adjacency_matrix_cpp(self):
        placedb = self.placedb
        start_time = tic()

        # 准备数据
        flat_netpin = placedb.flat_net2pin_map.astype(np.int32)
        netpin_start = placedb.flat_net2pin_start_map.astype(np.int32)
        pin2node_map = placedb.pin2node_map.astype(np.int32)
        net_weights = np.ones(placedb.num_nets, dtype=np.float32)
        net_mask = np.ones(placedb.num_nets, dtype=np.int32)

        # 调C++
        data, rows, cols = gift_adj_cpp.adj_matrix_forward(
            flat_netpin,
            netpin_start,
            pin2node_map,
            net_weights,
            net_mask,
            placedb.num_physical_nodes
        )

        self.adjacency_matrix = sp.coo_matrix(
            (data, (rows, cols)),
            shape=(placedb.num_physical_nodes, placedb.num_physical_nodes),
            dtype=np.float32
        ).tocsr()

        build_time = time.time() - start_time
        density = self.adjacency_matrix.nnz / (placedb.num_physical_nodes ** 2) * 100
        logger.info(f"C++邻接矩阵构建完成，耗时: {build_time:.3f}s, 非零元素: {self.adjacency_matrix.nnz:,}, 密度: {density:.4f}%")

    def build_adjacency_matrix_python(self):
        start_time = tic()
        placedb = self.placedb
        n = placedb.num_physical_nodes
        edge_weights = defaultdict(float)

        for net_id in range(placedb.num_nets):
            pin_start = placedb.flat_net2pin_start_map[net_id]
            pin_end = placedb.flat_net2pin_start_map[net_id + 1]
            num_pins = pin_end - pin_start
            if num_pins < 2:
                continue

            weight = 2.0 / num_pins

            # 收集节点
            nodes = []
            for j in range(pin_start, pin_end):
                flat_pin_id = placedb.flat_net2pin_map[j]
                node_id = placedb.pin2node_map[flat_pin_id]
                if node_id < n:
                    nodes.append(node_id)

            # clique模型
            for i in range(len(nodes)):
                for j in range(i + 1, len(nodes)):
                    edge = (min(nodes[i], nodes[j]), max(nodes[i], nodes[j]))
                    edge_weights[edge] += weight

        # 构建COO矩阵
        rows, cols, weights = [], [], []
        for (i, j), w in edge_weights.items():
            rows.extend([i, j])
            cols.extend([j, i])
            weights.extend([w, w])

        self.adjacency_matrix = sp.coo_matrix(
            (weights, (rows, cols)),
            shape=(n, n),
            dtype=np.float32
        ).tocsr()

        build_time = time.time() - start_time
        density = self.adjacency_matrix.nnz / (n * n) * 100
        logger.info(f"Python邻接矩阵构建完成，耗时: {build_time:.3f}s, 非零元素: {self.adjacency_matrix.nnz:,}, 密度: {density:.4f}%")

    # ====== 初始化位置 ======
    def set_preset_center(self, center_x, center_y):
        self._preset_center = (center_x, center_y)
        logger.info(f"使用预设中心位置: ({center_x:.2f}, {center_y:.2f})")

    def initialize_positions(self):
        t0 = tic()
        placedb = self.placedb
        n = placedb.num_physical_nodes

        # 中心
        if hasattr(self, '_preset_center'):
            initLocX, initLocY = self._preset_center
        else:
            initLocX, initLocY = PlacementUtils.calculate_initial_center(placedb)

        # 初始位置数组
        initial_positions = np.zeros((n, 2), dtype=np.float32)

        # 固定节点位置（中心点）
        for i in range(placedb.num_movable_nodes, n):
            initial_positions[i, 0] = placedb.node_x[i] + placedb.node_size_x[i] / 2
            initial_positions[i, 1] = placedb.node_y[i] + placedb.node_size_y[i] / 2

        # 随机初始（按DREAMPlace风格）
        fixed_positions = initial_positions[placedb.num_movable_nodes:]
        random_initial = timed_step(
            "生成随机初始位置(generate_initial_locations)",
            self.generate_initial_locations, fixed_positions, placedb.num_movable_nodes, self.scale
        )
        initial_positions[:placedb.num_movable_nodes] = random_initial

        self.initial_positions = initial_positions
        toc(t0, "initialize_positions")
        return initial_positions

    def generate_initial_locations(self, fixed_cell_location, movable_num, scale):
        if len(fixed_cell_location) == 0:
            # 没有固定单元：使用预设中心或几何中心
            if hasattr(self, '_preset_center'):
                xcenter, ycenter = self._preset_center
            else:
                xcenter = (self.xh + self.xl) / 2
                ycenter = (self.yh + self.yl) / 2
            x_range = self.xh - self.xl
            y_range = self.yh - self.yl
        else:
            # 有固定单元：先计算边界（只算一次）
            x_min, x_max = np.min(fixed_cell_location[:, 0]), np.max(fixed_cell_location[:, 0])
            y_min, y_max = np.min(fixed_cell_location[:, 1]), np.max(fixed_cell_location[:, 1])
            x_range = x_max - x_min
            y_range = y_max - y_min
            
            # 然后决定中心位置
            if hasattr(self, '_preset_center'):
                xcenter, ycenter = self._preset_center  # 使用更精确的预设中心
            else:
                xcenter = (x_max + x_min) / 2  # 基于已计算的边界
                ycenter = (y_max + y_min) / 2

        # 生成随机分布
        random_initial = np.random.rand(int(movable_num), 2)
        random_initial[:, 0] = ((random_initial[:, 0] - 0.5) * x_range * scale) + xcenter
        random_initial[:, 1] = ((random_initial[:, 1] - 0.5) * y_range * scale) + ycenter
        return random_initial

    # ====== GiFt 滤波 ======
# ====== GiFt 滤波 ======
    def apply_gift_filters(self, initial_positions):
        """
        GPU传输和批量处理优化版本
        """
        logger.info("应用GiFt滤波器...")
        if self.adjacency_matrix is None:
            self.build_adjacency_matrix()

        # 优化1: 异步GPU数据传输
        t_to_torch = tic()
        if self.device.type == 'cuda':
            try:
                # 使用pinned memory + 异步传输
                cpu_tensor = torch.from_numpy(initial_positions).float().pin_memory()
                random_initial = cpu_tensor.to(self.device, non_blocking=True)
            except Exception as e:
                logger.warning(f"异步传输失败，回退到同步: {e}")
                random_initial = torch.from_numpy(initial_positions).float().to(self.device)
        else:
            random_initial = torch.from_numpy(initial_positions).float().to(self.device)
        toc(t_to_torch, "初始位置 转换为 Torch 张量")

        # 优化2: 复用GPU滤波器实例
        if not hasattr(self, '_optimized_gpu_filter'):
            self._optimized_gpu_filter = GiFtGPUFilter(self.adjacency_matrix, self.device)
        
        gpu_filter = self._optimized_gpu_filter

        # 优化3: 批量预训练所有σ值
        gpu_filter.batch_train([4, 2])  # 预先计算所有需要的σ值
        
        # 低通滤波：σ=4, k=4
        t_low = tic()
        location_low = gpu_filter.get_cell_position(4, random_initial, sigma=4)
        toc(t_low, "低通滤波 (σ=4, k=4)")

        # 中通滤波：σ=4, k=2（复用σ=4）
        t_mid = tic()
        location_m = gpu_filter.get_cell_position(2, random_initial, sigma=4)
        toc(t_mid, "中通滤波 (σ=4, k=2)")

        # 高通滤波：σ=2, k=2
        t_high = tic()
        location_h = gpu_filter.get_cell_position(2, random_initial, sigma=2)
        toc(t_high, "高通滤波 (σ=2, k=2)")

        # 优化4: 滤波结果组合
        t_combine = tic()
        location = 0.2 * location_low + 0.7 * location_m + 0.1 * location_h
        toc(t_combine, "滤波结果组合")

        # 优化5: 优化数据回传
        t_back = tic()
        if self.device.type == 'cuda':
            torch.cuda.synchronize()  # 确保GPU计算完成
        optimized_positions = location.cpu().numpy()
        toc(t_back, "Torch 张量 转回 Numpy")

        # 固定节点位置保持不变
        num_movable = self.placedb.num_movable_nodes
        optimized_positions[num_movable:] = initial_positions[num_movable:]

        return optimized_positions

    # ====== 约束 ======
    def apply_placement_constraints(self, positions):
        """
        应用布局约束 - 向量化版本
        """
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
            
            # 批量计算半宽半高
            half_widths = placedb.node_size_x[:num_movable] / 2
            half_heights = placedb.node_size_y[:num_movable] / 2
            
            # 计算边界
            x_min_bounds = placedb.xl + half_widths
            x_max_bounds = placedb.xh - half_widths
            y_min_bounds = placedb.yl + half_heights
            y_max_bounds = placedb.yh - half_heights
            
            # 向量化约束 - 一次性处理所有可移动节点
            constrained_positions[:num_movable, 0] = np.clip(
                positions[:num_movable, 0], x_min_bounds, x_max_bounds
            )
            constrained_positions[:num_movable, 1] = np.clip(
                positions[:num_movable, 1], y_min_bounds, y_max_bounds
            )
            
            toc(boundary_t0, "边界约束(向量化)")
        
        # 资源约束 - 保持原有逻辑（如果启用）
        if self.enable_resource_constraints and self.resource_regions:
            resource_t0 = tic()
            
            # 这里仍使用原有的逐节点处理逻辑
            # 因为资源约束涉及复杂的区域查找，向量化较困难
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
        # 已在区域内
        for region in regions:
            if (region[0] + half_width <= center_x <= region[2] - half_width and
                region[1] + half_height <= center_y <= region[3] - half_height):
                return center_x, center_y

        # 投影到最近的区域
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

    # ====== 主流程 ======
    def optimize_placement(self):
        total_start = tic()

        # 初始化位置
        if self.initial_positions is None:
            self.initial_positions = timed_step("初始化位置(initialize_positions)", self.initialize_positions)

        # 应用GiFt算法
        self.optimized_positions = timed_step("应用GiFt滤波器(apply_gift_filters)",
                                             self.apply_gift_filters, self.initial_positions)

        # 应用布局约束
        self.optimized_positions = timed_step("应用布局约束(apply_placement_constraints)",
                                             self.apply_placement_constraints, self.optimized_positions)

        # HPWL（可开关 + 可选后端）
        if self.enable_hpwl:
            if self.hpwl_backend == 'numpy':
                initial_wl = timed_step("初始HPWL计算(calculate_hpwl_numpy)", self.calculate_hpwl_numpy, self.initial_positions)
                optimized_wl = timed_step("优化后HPWL计算(calculate_hpwl_numpy)", self.calculate_hpwl_numpy, self.optimized_positions)
            else:  # 'naive' 默认保持原实现
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

    # ====== 输出给DREAMPlace ======
    def get_dreamplace_positions(self):
        if self.optimized_positions is None:
            self.optimize_placement()

        placedb = self.placedb
        pos = np.zeros(placedb.num_nodes * 2, dtype=placedb.dtype)

        # 复制现有位置
        pos[0:placedb.num_physical_nodes] = placedb.node_x
        pos[placedb.num_nodes:placedb.num_nodes+placedb.num_physical_nodes] = placedb.node_y

        # 更新可移动节点（左下角坐标）
        for i in range(placedb.num_movable_nodes):
            center_x = self.optimized_positions[i, 0]
            center_y = self.optimized_positions[i, 1]
            pos[i] = center_x - placedb.node_size_x[i] / 2
            pos[placedb.num_nodes + i] = center_y - placedb.node_size_y[i] / 2

        return pos

    # ====== HPWL（原版：逐网循环，结果与旧版完全一致） ======
    def calculate_hpwl(self, positions):
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

    # ====== HPWL（可选：NumPy分段向量化版，保持与原逻辑一致的条件判断） ======
    def calculate_hpwl_numpy(self, positions):
        """
        更快的 NumPy 实现，保持“每网至少两个pin，且 x/y 范围都 > 0 才计入”的逻辑。
        """
        t0 = tic()
        placedb = self.placedb

        # 展开到引脚级
        flat_pin = placedb.flat_net2pin_map
        pin2node = placedb.pin2node_map
        node_id = pin2node[flat_pin]  # 长度 = 总pins
        # 取中心坐标（按pin映射）
        cx = positions[node_id, 0]
        cy = positions[node_id, 1]
        # 尺寸、偏移（按pin映射）
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

        # 仅对有效段做 reduceat
        s = starts[:-1][valid]
        e = starts[1:][valid]

        xmin = np.minimum.reduceat(px, s)
        xmax = np.maximum.reduceat(px, s)
        ymin = np.minimum.reduceat(py, s)
        ymax = np.maximum.reduceat(py, s)

        # 按原逻辑：仅当 x_max > x_min 且 y_max > y_min 才计入
        rx = xmax - xmin
        ry = ymax - ymin
        positive = (rx > 0) & (ry > 0)
        hpwl = (rx + ry) * positive

        total = float(hpwl.sum())
        toc(t0, "calculate_hpwl_numpy")
        return total

    # ====== 可视化 ======
    def visualize_placement(self, output_file=None, show_nets=False):
        if self.optimized_positions is None:
            logger.warning("没有优化位置数据，无法进行可视化")
            return
        return timed_step("可视化(optimized)", self.visualize_placement_with_positions,
                          self.optimized_positions, output_file, "GiFt Placement Result", show_nets)

    def visualize_placement_with_positions(self, positions, output_file=None, title="GiFt Placement", show_nets=False):
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

