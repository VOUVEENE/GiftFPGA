#!/usr/bin/env python3
"""
GiFT问题诊断脚本
用于验证坐标系转换和HPWL计算的正确性
"""

import numpy as np
import matplotlib.pyplot as plt
import logging
from gift_init_placer import GiFtFPGAPlacer
from placement_utils import PlacementUtils

class GiFTDiagnostic:
    def __init__(self, placedb, params):
        self.placedb = placedb
        self.params = params
        
    def test_coordinate_systems(self):
        """测试坐标系转换的正确性"""
        print("="*60)
        print("坐标系转换诊断")
        print("="*60)
        
        # 创建简单的测试位置
        test_positions_center = np.array([
            [10.0, 20.0],  # 节点0中心位置
            [30.0, 40.0],  # 节点1中心位置
            [50.0, 60.0],  # 节点2中心位置
        ])
        
        # 模拟节点尺寸
        node_sizes_x = np.array([2.0, 4.0, 6.0])
        node_sizes_y = np.array([3.0, 5.0, 7.0])
        
        print("测试数据:")
        for i in range(3):
            center_x, center_y = test_positions_center[i]
            size_x, size_y = node_sizes_x[i], node_sizes_y[i]
            
            # 手动计算左下角坐标
            left_x = center_x - size_x / 2
            bottom_y = center_y - size_y / 2
            
            print(f"节点{i}: 中心=({center_x:.1f}, {center_y:.1f}), "
                  f"尺寸=({size_x:.1f}x{size_y:.1f}), "
                  f"左下角=({left_x:.1f}, {bottom_y:.1f})")
        
        return test_positions_center, node_sizes_x, node_sizes_y
    
    def test_hpwl_calculation(self):
        """测试HPWL计算的一致性"""
        print("\n" + "="*60)
        print("HPWL计算一致性测试")
        print("="*60)
        
        # 创建GiFT实例
        gift_placer = GiFtFPGAPlacer(self.placedb, self.params)
        
        # 生成初始位置
        initial_positions = gift_placer.initialize_positions()
        
        # 使用两种方法计算HPWL
        try:
            hpwl_naive = gift_placer.calculate_hpwl(initial_positions)
            print(f"原版HPWL计算: {hpwl_naive:.2f}")
        except Exception as e:
            print(f"原版HPWL计算失败: {e}")
            hpwl_naive = None
            
        try:
            hpwl_numpy = gift_placer.calculate_hpwl_numpy(initial_positions)
            print(f"NumPy HPWL计算: {hpwl_numpy:.2f}")
        except Exception as e:
            print(f"NumPy HPWL计算失败: {e}")
            hpwl_numpy = None
        
        if hpwl_naive is not None and hpwl_numpy is not None:
            diff = abs(hpwl_naive - hpwl_numpy)
            rel_diff = diff / max(hpwl_naive, hpwl_numpy) * 100
            print(f"差异: {diff:.2f} ({rel_diff:.2f}%)")
            
            if rel_diff < 0.01:
                print("✓ HPWL计算一致性良好")
            else:
                print("✗ HPWL计算存在差异")
        
        return initial_positions
    
    def test_dreamplace_conversion(self):
        """测试转换为DREAMPlace格式的正确性"""
        print("\n" + "="*60)
        print("DREAMPlace格式转换测试")
        print("="*60)
        
        gift_placer = GiFtFPGAPlacer(self.placedb, self.params)
        
        # 创建已知的中心位置
        test_center_positions = np.zeros((self.placedb.num_physical_nodes, 2))
        
        # 只设置前几个可移动节点的位置
        for i in range(min(5, self.placedb.num_movable_nodes)):
            test_center_positions[i, 0] = 100.0 + i * 10  # x坐标
            test_center_positions[i, 1] = 200.0 + i * 15  # y坐标
        
        # 设置固定节点位置（保持原有位置）
        for i in range(self.placedb.num_movable_nodes, self.placedb.num_physical_nodes):
            test_center_positions[i, 0] = self.placedb.node_x[i] + self.placedb.node_size_x[i] / 2
            test_center_positions[i, 1] = self.placedb.node_y[i] + self.placedb.node_size_y[i] / 2
        
        gift_placer.optimized_positions = test_center_positions
        
        # 转换为DREAMPlace格式
        dreamplace_pos = gift_placer.get_dreamplace_positions()
        
        print("坐标转换验证（前5个可移动节点）:")
        for i in range(min(5, self.placedb.num_movable_nodes)):
            center_x = test_center_positions[i, 0]
            center_y = test_center_positions[i, 1]
            
            dreamplace_x = dreamplace_pos[i]
            dreamplace_y = dreamplace_pos[self.placedb.num_nodes + i]
            
            expected_x = center_x - self.placedb.node_size_x[i] / 2
            expected_y = center_y - self.placedb.node_size_y[i] / 2
            
            print(f"节点{i}:")
            print(f"  中心位置: ({center_x:.2f}, {center_y:.2f})")
            print(f"  转换结果: ({dreamplace_x:.2f}, {dreamplace_y:.2f})")
            print(f"  期望结果: ({expected_x:.2f}, {expected_y:.2f})")
            print(f"  误差: ({abs(dreamplace_x - expected_x):.4f}, {abs(dreamplace_y - expected_y):.4f})")
            
            if abs(dreamplace_x - expected_x) < 1e-6 and abs(dreamplace_y - expected_y) < 1e-6:
                print("  ✓ 转换正确")
            else:
                print("  ✗ 转换错误")
            print()
        
        return dreamplace_pos
    
    def compare_initial_positions(self):
        """比较不同初始化方法的位置分布"""
        print("\n" + "="*60)
        print("初始位置分布比较")
        print("="*60)
        
        # 计算标准初始中心
        initLocX, initLocY = PlacementUtils.calculate_initial_center(self.placedb)
        print(f"计算的初始中心: ({initLocX:.2f}, {initLocY:.2f})")
        
        # 获取DREAMPlace默认初始化位置（左下角坐标）
        default_init_x = self.placedb.node_x[:self.placedb.num_movable_nodes].copy()
        default_init_y = self.placedb.node_y[:self.placedb.num_movable_nodes].copy()
        
        # 转换为中心坐标
        default_center_x = default_init_x + self.placedb.node_size_x[:self.placedb.num_movable_nodes] / 2
        default_center_y = default_init_y + self.placedb.node_size_y[:self.placedb.num_movable_nodes] / 2
        
        # 创建GiFT初始化
        gift_placer = GiFtFPGAPlacer(self.placedb, self.params)
        gift_placer.set_preset_center(initLocX, initLocY)
        gift_positions = gift_placer.initialize_positions()
        
        print("位置分布统计（前1000个可移动节点）:")
        n_sample = min(1000, self.placedb.num_movable_nodes)
        
        print(f"默认初始化中心位置:")
        print(f"  X: 均值={default_center_x[:n_sample].mean():.2f}, 标准差={default_center_x[:n_sample].std():.2f}")
        print(f"  Y: 均值={default_center_y[:n_sample].mean():.2f}, 标准差={default_center_y[:n_sample].std():.2f}")
        
        print(f"GiFT初始化中心位置:")
        print(f"  X: 均值={gift_positions[:n_sample, 0].mean():.2f}, 标准差={gift_positions[:n_sample, 0].std():.2f}")
        print(f"  Y: 均值={gift_positions[:n_sample, 1].mean():.2f}, 标准差={gift_positions[:n_sample, 1].std():.2f}")
        
        return default_center_x, default_center_y, gift_positions
    
    def visualize_positions(self, default_center_x, default_center_y, gift_positions, sample_size=1000):
        """可视化位置分布"""
        print("\n生成位置分布可视化...")
        
        n_sample = min(sample_size, self.placedb.num_movable_nodes)
        
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))
        
        # 默认初始化分布
        ax1.scatter(default_center_x[:n_sample], default_center_y[:n_sample], 
                   alpha=0.6, s=1, c='blue')
        ax1.set_title('默认初始化位置分布')
        ax1.set_xlabel('X坐标')
        ax1.set_ylabel('Y坐标')
        ax1.grid(True, alpha=0.3)
        
        # GiFT初始化分布
        ax2.scatter(gift_positions[:n_sample, 0], gift_positions[:n_sample, 1], 
                   alpha=0.6, s=1, c='red')
        ax2.set_title('GiFT初始化位置分布')
        ax2.set_xlabel('X坐标')
        ax2.set_ylabel('Y坐标')
        ax2.grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.savefig('position_distribution_comparison.png', dpi=300, bbox_inches='tight')
        print("位置分布图已保存为 'position_distribution_comparison.png'")
        plt.show()
    
    def run_full_diagnostic(self):
        """运行完整的诊断流程"""
        print("开始GiFT问题全面诊断...")
        
        # 1. 坐标系转换测试
        test_pos, size_x, size_y = self.test_coordinate_systems()
        
        # 2. HPWL计算测试
        initial_pos = self.test_hpwl_calculation()
        
        # 3. DREAMPlace转换测试
        dreamplace_pos = self.test_dreamplace_conversion()
        
        # 4. 位置分布比较
        default_x, default_y, gift_pos = self.compare_initial_positions()
        
        # 5. 可视化
        self.visualize_positions(default_x, default_y, gift_pos)
        
        print("\n" + "="*60)
        print("诊断完成")
        print("="*60)

# 使用示例
def run_diagnostic(placedb, params):
    """运行诊断的主函数"""
    diagnostic = GiFTDiagnostic(placedb, params)
    diagnostic.run_full_diagnostic()

if __name__ == "__main__":
    # 需要从主程序传入 placedb 和 params
    print("请在主程序中调用 run_diagnostic(placedb, params)")