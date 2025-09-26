#!/usr/bin/env python3
"""
FPGA HPWL Results Comparison Script - 无第三方依赖版本
专门分析HPWL最终结果、迭代轮次、时间对比，以no_gift为基准
"""

import os
import re
import json
import statistics
from pathlib import Path
import argparse
from datetime import datetime

class SimpleFPGAAnalyzer:
    def __init__(self, history_dir="history_results"):
        self.history_dir = Path(history_dir)
        
    def parse_log_metrics(self, log_path):
        """提取关键指标"""
        if not os.path.exists(log_path):
            return None
            
        with open(log_path, 'r', encoding='utf-8') as f:
            content = f.read()
            
        metrics = {}
        
        # 1. 最终HPWL (从最后一次迭代提取)
        iter_pattern = r'iter:\s*(\d+),\s*HPWL\s+([\d\.E\+\-]+),'
        all_iters = re.findall(iter_pattern, content)
        if all_iters:
            metrics['final_hpwl'] = float(all_iters[-1][1])
            metrics['iterations'] = len(all_iters)
        else:
            metrics['final_hpwl'] = None
            metrics['iterations'] = 0
            
        # 2. 总体布局时间
        time_pattern = r'Placement completed in ([\d\.]+) seconds'
        time_match = re.search(time_pattern, content)
        metrics['placement_time'] = float(time_match.group(1)) if time_match else None
        
        # 3. GiFt相关时间 (仅with_gift有)
        gift_time_pattern = r'GiFt优化完成，总耗时:\s*([\d\.]+)s'
        gift_time_match = re.search(gift_time_pattern, content)
        metrics['gift_time'] = float(gift_time_match.group(1)) if gift_time_match else 0
        
        # 4. 邻接矩阵构建时间
        adj_matrix_pattern = r'邻接矩阵构建完成，耗时:\s*([\d\.]+)s'
        adj_matrix_match = re.search(adj_matrix_pattern, content)
        metrics['adj_matrix_time'] = float(adj_matrix_match.group(1)) if adj_matrix_match else 0
        
        return metrics
    
    def analyze_all_benchmarks(self, version="v3_1"):
        """分析所有基准测试"""
        version_path = self.history_dir / version
        
        if not version_path.exists():
            print(f"版本 {version} 不存在!")
            return []
        
        # 获取所有基准测试
        results_path = version_path / "results"
        benchmarks = []
        if results_path.exists():
            benchmarks = [d.name for d in results_path.iterdir() if d.is_dir()]
        
        comparison_data = []
        
        for benchmark in benchmarks:
            print(f"分析 {benchmark}...")
            
            # with GiFt 路径
            gift_log = version_path / "results" / benchmark / f"{benchmark}.log"
            # without GiFt 路径  
            no_gift_log = version_path / "results_no_gift" / benchmark / f"{benchmark}_no_gift.log"
            
            if not gift_log.exists():
                gift_log = version_path / "results" / benchmark / f"{benchmark}_with_gift.log"
            
            if not gift_log.exists() or not no_gift_log.exists():
                print(f"  跳过 {benchmark}: 日志文件不完整")
                continue
                
            gift_metrics = self.parse_log_metrics(gift_log)
            no_gift_metrics = self.parse_log_metrics(no_gift_log)
            
            if not gift_metrics or not no_gift_metrics:
                print(f"  跳过 {benchmark}: 无法解析指标")
                continue
                
            if not gift_metrics['final_hpwl'] or not no_gift_metrics['final_hpwl']:
                print(f"  跳过 {benchmark}: HPWL数据缺失")
                continue
            
            # 计算改进率 (以no_gift为基准)
            hpwl_improvement = ((no_gift_metrics['final_hpwl'] - gift_metrics['final_hpwl']) / 
                               no_gift_metrics['final_hpwl'] * 100)
            
            comparison_data.append({
                'Benchmark': benchmark,
                'HPWL_No_GiFt': no_gift_metrics['final_hpwl'],
                'HPWL_With_GiFt': gift_metrics['final_hpwl'],
                'HPWL_Improvement_%': hpwl_improvement,
                'Iterations_No_GiFt': no_gift_metrics['iterations'],
                'Iterations_With_GiFt': gift_metrics['iterations'],
                'Time_No_GiFt_s': no_gift_metrics['placement_time'] or 0,
                'Time_With_GiFt_s': gift_metrics['placement_time'] or 0,
                'GiFt_Opt_Time_s': gift_metrics['gift_time'],
                'Adj_Matrix_Time_s': gift_metrics['adj_matrix_time']
            })
        
        return comparison_data
    
    def filter_outliers(self, data, threshold_percent=50):
        """剔除结果差别过大的例子"""
        if not data:
            return data, []
            
        filtered_data = []
        removed_benchmarks = []
        
        for item in data:
            abs_improvement = abs(item['HPWL_Improvement_%'])
            if abs_improvement <= threshold_percent:
                filtered_data.append(item)
            else:
                removed_benchmarks.append(item['Benchmark'])
        
        return filtered_data, removed_benchmarks
    
    def write_csv(self, data, filepath):
        """写入CSV文件"""
        if not data:
            return
            
        headers = [
            'Benchmark', 'HPWL_No_GiFt', 'HPWL_With_GiFt', 'HPWL_Improvement_%',
            'Iterations_No_GiFt', 'Iterations_With_GiFt', 'Time_No_GiFt_s', 
            'Time_With_GiFt_s', 'GiFt_Opt_Time_s', 'Adj_Matrix_Time_s'
        ]
        
        with open(filepath, 'w', encoding='utf-8') as f:
            # 写入表头
            f.write(','.join(headers) + '\n')
            
            # 写入数据
            for item in data:
                row = [str(item[h]) for h in headers]
                f.write(','.join(row) + '\n')
    
    def print_table(self, data):
        """打印格式化表格"""
        if not data:
            return
            
        # 计算列宽
        headers = [
            'Benchmark', 'HPWL_No_GiFt', 'HPWL_With_GiFt', 'HPWL_Imp_%',
            'Iter_No', 'Iter_With', 'Time_No', 'Time_With', 'GiFt_Time', 'Adj_Time'
        ]
        
        col_widths = [len(h) for h in headers]
        
        for item in data:
            col_widths[0] = max(col_widths[0], len(item['Benchmark']))
            col_widths[1] = max(col_widths[1], len(f"{item['HPWL_No_GiFt']:.0f}"))
            col_widths[2] = max(col_widths[2], len(f"{item['HPWL_With_GiFt']:.0f}"))
            col_widths[3] = max(col_widths[3], len(f"{item['HPWL_Improvement_%']:.2f}"))
            col_widths[4] = max(col_widths[4], len(str(item['Iterations_No_GiFt'])))
            col_widths[5] = max(col_widths[5], len(str(item['Iterations_With_GiFt'])))
            col_widths[6] = max(col_widths[6], len(f"{item['Time_No_GiFt_s']:.1f}"))
            col_widths[7] = max(col_widths[7], len(f"{item['Time_With_GiFt_s']:.1f}"))
            col_widths[8] = max(col_widths[8], len(f"{item['GiFt_Opt_Time_s']:.2f}"))
            col_widths[9] = max(col_widths[9], len(f"{item['Adj_Matrix_Time_s']:.2f}"))
        
        # 打印表头
        header_line = ""
        for i, h in enumerate(headers):
            header_line += h.ljust(col_widths[i] + 2)
        print(header_line)
        print("-" * len(header_line))
        
        # 打印数据行
        for item in data:
            line = f"{item['Benchmark'].ljust(col_widths[0] + 2)}"
            line += f"{item['HPWL_No_GiFt']:.0f}".ljust(col_widths[1] + 2)
            line += f"{item['HPWL_With_GiFt']:.0f}".ljust(col_widths[2] + 2)
            line += f"{item['HPWL_Improvement_%']:.2f}".ljust(col_widths[3] + 2)
            line += f"{item['Iterations_No_GiFt']}".ljust(col_widths[4] + 2)
            line += f"{item['Iterations_With_GiFt']}".ljust(col_widths[5] + 2)
            line += f"{item['Time_No_GiFt_s']:.1f}".ljust(col_widths[6] + 2)
            line += f"{item['Time_With_GiFt_s']:.1f}".ljust(col_widths[7] + 2)
            line += f"{item['GiFt_Opt_Time_s']:.2f}".ljust(col_widths[8] + 2)
            line += f"{item['Adj_Matrix_Time_s']:.2f}".ljust(col_widths[9] + 2)
            print(line)
    
    def generate_summary_table(self, version="v3_1", filter_outliers=True, outlier_threshold=50, output_dir=None):
        """生成汇总表格"""
        data = self.analyze_all_benchmarks(version)
        
        if not data:
            print("没有找到可分析的数据!")
            return None
        
        # 设置输出目录
        if output_dir is None:
            output_dir = self.history_dir / version
        else:
            output_dir = Path(output_dir)
        
        output_dir.mkdir(exist_ok=True)
        
        print(f"\n原始数据: {len(data)} 个基准测试")
        
        removed_benchmarks = []
        # 剔除异常值
        if filter_outliers:
            data, removed_benchmarks = self.filter_outliers(data, outlier_threshold)
            if removed_benchmarks:
                print(f"剔除异常结果 (HPWL改进绝对值 > {outlier_threshold}%): {removed_benchmarks}")
            print(f"过滤后数据: {len(data)} 个基准测试")
        
        if not data:
            print("过滤后没有剩余数据!")
            return None
        
        # 按HPWL改进排序
        data.sort(key=lambda x: x['HPWL_Improvement_%'], reverse=True)
        
        # 计算统计数据
        improvements = [item['HPWL_Improvement_%'] for item in data]
        times_no_gift = [item['Time_No_GiFt_s'] for item in data if item['Time_No_GiFt_s'] > 0]
        times_with_gift = [item['Time_With_GiFt_s'] for item in data if item['Time_With_GiFt_s'] > 0]
        gift_times = [item['GiFt_Opt_Time_s'] for item in data]
        adj_times = [item['Adj_Matrix_Time_s'] for item in data]
        iter_no_gift = [item['Iterations_No_GiFt'] for item in data]
        iter_with_gift = [item['Iterations_With_GiFt'] for item in data]
        
        positive_improvements = len([x for x in improvements if x > 0])
        
        # 显示汇总统计
        print("\n" + "="*80)
        print("FPGA HPWL 结果对比汇总 (以 No_GiFt 为基准)")
        print("="*80)
        
        print(f"\n基准测试数量: {len(data)}")
        print(f"平均HPWL改进: {statistics.mean(improvements):.2f}%")
        print(f"中位数HPWL改进: {statistics.median(improvements):.2f}%")
        if len(improvements) > 1:
            print(f"标准差: {statistics.stdev(improvements):.2f}%")
        print(f"最大改进: {max(improvements):.2f}%")
        print(f"最小改进: {min(improvements):.2f}%")
        print(f"正向改进的基准: {positive_improvements}/{len(data)} ({positive_improvements/len(data)*100:.1f}%)")
        
        # 时间统计
        if times_no_gift and times_with_gift:
            print(f"\n时间统计:")
            print(f"平均布局时间 (No GiFt): {statistics.mean(times_no_gift):.2f}s")
            print(f"平均布局时间 (With GiFt): {statistics.mean(times_with_gift):.2f}s")
        if gift_times:
            print(f"平均GiFt优化时间: {statistics.mean(gift_times):.2f}s")
        if adj_times:
            print(f"平均邻接矩阵构建时间: {statistics.mean(adj_times):.2f}s")
        
        # 迭代次数统计
        print(f"\n迭代次数统计:")
        print(f"平均迭代次数 (No GiFt): {statistics.mean(iter_no_gift):.1f}")
        print(f"平均迭代次数 (With GiFt): {statistics.mean(iter_with_gift):.1f}")
        
        print("\n" + "="*80)
        print("详细结果表格:")
        print("="*80)
        
        self.print_table(data)
        
        # 生成文件名
        suffix = "filtered" if filter_outliers else "full"
        csv_filename = f"hpwl_comparison_{version}_{suffix}.csv"
        report_filename = f"hpwl_analysis_report_{version}_{suffix}.txt"
        
        # 保存到CSV
        csv_path = output_dir / csv_filename
        self.write_csv(data, csv_path)
        print(f"\nCSV结果已保存到: {csv_path}")
        
        # 生成详细报告文件
        report_path = output_dir / report_filename
        with open(report_path, 'w', encoding='utf-8') as f:
            f.write(f"FPGA HPWL 结果对比分析报告 - 版本 {version}\n")
            f.write("=" * 60 + "\n\n")
            f.write(f"分析时间: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
            f.write(f"基准测试数量: {len(data)}\n")
            f.write(f"异常值过滤: {'是' if filter_outliers else '否'} (阈值: {outlier_threshold}%)\n")
            if removed_benchmarks:
                f.write(f"被剔除的基准: {', '.join(removed_benchmarks)}\n")
            f.write("\n")
            
            f.write("汇总统计:\n")
            f.write("-" * 20 + "\n")
            f.write(f"平均HPWL改进: {statistics.mean(improvements):.2f}%\n")
            f.write(f"中位数HPWL改进: {statistics.median(improvements):.2f}%\n")
            if len(improvements) > 1:
                f.write(f"标准差: {statistics.stdev(improvements):.2f}%\n")
            f.write(f"最大改进: {max(improvements):.2f}%\n")
            f.write(f"最小改进: {min(improvements):.2f}%\n")
            f.write(f"正向改进的基准: {positive_improvements}/{len(data)} ({positive_improvements/len(data)*100:.1f}%)\n\n")
            
            if times_no_gift and times_with_gift:
                f.write("时间统计:\n")
                f.write("-" * 20 + "\n")
                f.write(f"平均布局时间 (No GiFt): {statistics.mean(times_no_gift):.2f}s\n")
                f.write(f"平均布局时间 (With GiFt): {statistics.mean(times_with_gift):.2f}s\n")
                if gift_times:
                    f.write(f"平均GiFt优化时间: {statistics.mean(gift_times):.2f}s\n")
                if adj_times:
                    f.write(f"平均邻接矩阵构建时间: {statistics.mean(adj_times):.2f}s\n")
                f.write("\n")
            
            f.write("详细数据:\n")
            f.write("-" * 20 + "\n")
            for item in data:
                f.write(f"{item['Benchmark']:<15} HPWL: {item['HPWL_No_GiFt']:>8.0f} -> {item['HPWL_With_GiFt']:>8.0f} ({item['HPWL_Improvement_%']:>6.2f}%) ")
                f.write(f"Iter: {item['Iterations_No_GiFt']:>3} -> {item['Iterations_With_GiFt']:>3} ")
                f.write(f"Time: {item['Time_No_GiFt_s']:>5.1f}s -> {item['Time_With_GiFt_s']:>5.1f}s\n")
        
        print(f"详细报告已保存到: {report_path}")
        print(f"\n所有文件都保存在目录: {output_dir}")
        
        return data

def main():
    parser = argparse.ArgumentParser(description="FPGA HPWL结果对比分析 - 无依赖版本")
    parser.add_argument("--history_dir", default="../history_results", 
                       help="历史结果目录路径")
    parser.add_argument("--version", default="v3_1", 
                       help="要分析的版本")
    parser.add_argument("--no_filter", action="store_true",
                       help="不过滤异常值")
    parser.add_argument("--outlier_threshold", type=float, default=50.0,
                       help="异常值阈值 (HPWL改进绝对值百分比)")
    parser.add_argument("--output_dir", 
                       help="输出目录 (默认为版本目录)")
    
    args = parser.parse_args()
    
    analyzer = SimpleFPGAAnalyzer(args.history_dir)
    
    # 生成分析结果
    result = analyzer.generate_summary_table(
        version=args.version, 
        filter_outliers=not args.no_filter,
        outlier_threshold=args.outlier_threshold,
        output_dir=args.output_dir
    )
    
    if result is None:
        print("分析失败，没有生成结果文件。")

if __name__ == "__main__":
    main()