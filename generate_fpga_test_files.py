"""
FPGA测试文件生成脚本
根据FPGA01.json模板生成FPGA02到FPGA12的配置文件
"""

import json
import os

def generate_fpga_test_files():
    """生成FPGA02到FPGA12的测试配置文件"""
    
    # 基础模板配置
    base_config = {
        "gpu": 1,
        "num_bins_x": 512,
        "num_bins_y": 512,
        "global_place_stages": [
            {
                "num_bins_x": 512,
                "num_bins_y": 512,
                "iteration": 2000,
                "learning_rate": 0.01,
                "wirelength": "weighted_average",
                "optimizer": "nesterov"
            }
        ],
        "target_density": 1.0,
        "density_weight": 8e-5,
        "random_seed": 1000,
        "scale_factor": 1.0,
        "global_place_flag": 1,
        "legalize_flag": 0,
        "detailed_place_flag": 0,
        "dtype": "float32",
        "deterministic_flag": 0,
        "use_custom_init_place": 0,
        "use_gift_init_place": 1,
        "gift_alpha0": 0.1,
        "gift_alpha1": 0.7,
        "gift_alpha2": 0.2,
        "gift_enable_boundary_constraints": 1,
        "gift_enable_resource_constraints": 0,
        "plot_flag": 1,
        "log_to_file": 1,
        "_comment_network_analysis": "=== 网络分析相关参数 ===",
        "enable_network_analysis": True,
        "net_analysis_threshold": 50,
        "net_skip_threshold": 200,
        "net_analysis_export_detailed": True,
        "net_analysis_export_path": "./network_analysis_reports",
        "_comment_gift_optimization": "=== GiFt优化相关参数 ===",
        "gift_max_net_size": 1000,
        "gift_use_star_model": True,
        "gift_optimize_large_nets": True,
        "gift_sigma_2": 2,
        "gift_sigma_4": 4,
        "part_name": "xcvu095-ffva2104-2-e"
    }
    
    # 创建输出目录（如果不存在）
    output_dir = "test"
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
        print(f"创建输出目录: {output_dir}")
    
    # 生成FPGA02到FPGA12的配置文件
    for i in range(2, 13):  # 2到12
        fpga_num = f"{i:02d}"  # 格式化为两位数字，如02, 03, ..., 12
        
        # 复制基础配置
        config = base_config.copy()
        
        # 设置对应的aux_input路径
        config["aux_input"] = f"benchmarks/sample_ispd2016_benchmarks/FPGA{fpga_num}/design.aux"
        
        # 可以根据需要为不同的FPGA调整参数
        # 例如，为不同的FPGA设置不同的随机种子
        config["random_seed"] = 1000 + i
        
        # 生成文件名
        filename = os.path.join(output_dir, f"FPGA{fpga_num}.json")
        
        # 写入JSON文件
        with open(filename, 'w', encoding='utf-8') as f:
            json.dump(config, f, indent=4, ensure_ascii=False)
        
        print(f"已生成: {filename}")

def validate_generated_files():
    """验证生成的文件是否正确"""
    print("\n=== 验证生成的文件 ===")
    
    for i in range(2, 13):
        fpga_num = f"{i:02d}"
        filename = f"test/FPGA{fpga_num}.json"
        
        if os.path.exists(filename):
            with open(filename, 'r', encoding='utf-8') as f:
                try:
                    config = json.load(f)
                    aux_input = config.get("aux_input", "")
                    expected_aux = f"benchmarks/sample_ispd2016_benchmarks/FPGA{fpga_num}/design.aux"
                    
                    if aux_input == expected_aux:
                        print(f"✓ {filename}: aux_input路径正确")
                    else:
                        print(f"✗ {filename}: aux_input路径不匹配")
                        print(f"  期望: {expected_aux}")
                        print(f"  实际: {aux_input}")
                except json.JSONDecodeError:
                    print(f"✗ {filename}: JSON格式错误")
        else:
            print(f"✗ {filename}: 文件不存在")

def show_example_content():
    """显示生成的文件示例内容"""
    example_file = "test/FPGA02.json"
    if os.path.exists(example_file):
        print(f"\n=== {example_file} 内容示例 ===")
        with open(example_file, 'r', encoding='utf-8') as f:
            content = f.read()
            # 只显示前几行
            lines = content.split('\n')
            for line in lines[:10]:
                print(line)
            if len(lines) > 10:
                print("...")

if __name__ == "__main__":
    print("开始生成FPGA测试配置文件...")
    
    # 生成文件
    generate_fpga_test_files()
    
    # 验证文件
    validate_generated_files()
    
    # 显示示例内容
    show_example_content()
    
    print(f"\n✅ 完成！已生成FPGA02.json到FPGA12.json共{12-2+1}个文件")
    print("所有文件保存在 test/ 目录下")