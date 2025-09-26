#!/usr/bin/env python3
"""
FPGA Placement Results Analysis Script
Analyzes DREAMPlaceFPGA placement results comparing GiFT vs non-GiFT approaches
"""

import os
import re
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
import numpy as np

class FPGAResultsAnalyzer:
    def __init__(self, results_dir="results"):
        self.results_dir = Path(results_dir)
        self.data = []
        
    def parse_log_file(self, log_path):
        """Parse a single log file to extract key metrics"""
        try:
            with open(log_path, 'r') as f:
                content = f.read()
        except:
            return None
            
        result = {
            'benchmark': log_path.stem.replace('.log', ''),
            'gift_enabled': '_no_gift' not in str(log_path),
            'total_time': None,
            'placement_time': None,
            'final_hpwl': None,
            'final_overflow': [],
            'iterations': None,
            'gift_time': None,
            'reading_time': None
        }
        
        # Extract total completion time
        time_match = re.search(r'Completed Placement in ([\d.]+) seconds', content)
        if time_match:
            result['total_time'] = float(time_match.group(1))
            
        # Extract placement time
        place_match = re.search(r'Placement completed in ([\d.]+) seconds', content)
        if place_match:
            result['placement_time'] = float(place_match.group(1))
            
        # Extract reading time
        read_match = re.search(r'reading benchmark takes ([\d.]+) seconds', content)
        if read_match:
            result['reading_time'] = float(read_match.group(1))
            
        # Extract GiFT time if present
        gift_match = re.search(r'GiFt优化完成.*总耗时: ([\d.]+)s', content)
        if gift_match:
            result['gift_time'] = float(gift_match.group(1))
            
        # Extract final HPWL and overflow
        final_lines = content.strip().split('\n')[-10:]  # Last 10 lines
        for line in reversed(final_lines):
            if 'HPWL' in line and 'Overflow' in line:
                hpwl_match = re.search(r'HPWL ([\d.E+]+)', line)
                # More robust overflow pattern matching
                overflow_match = re.findall(r'([\d.]+E[+-]?\d+|[\d.]+)', line.split('Overflow')[1])
                
                if hpwl_match:
                    try:
                        result['final_hpwl'] = float(hpwl_match.group(1))
                    except ValueError:
                        result['final_hpwl'] = None
                        
                if overflow_match and len(overflow_match) >= 4:
                    try:
                        result['final_overflow'] = []
                        for x in overflow_match[:4]:
                            # Handle incomplete scientific notation
                            if 'E' in x and not ('E+' in x or 'E-' in x):
                                x = x + '+00'  # Assume E+00 if incomplete
                            result['final_overflow'].append(float(x))
                    except ValueError:
                        result['final_overflow'] = []
                break
                
        # Extract iteration count and convergence data
        iter_matches = re.findall(r'iter:\s*(\d+)', content)
        if iter_matches:
            result['iterations'] = int(iter_matches[-1]) + 1
            
        # Extract iteration convergence data
        result['convergence_data'] = self._extract_convergence_data(content)
            
        return result
    
    def _extract_convergence_data(self, content):
        """Extract iteration-by-iteration convergence data"""
        convergence_data = []
        
        # Find all iteration lines with HPWL and Overflow data
        iter_pattern = r'iter:\s*(\d+),\s*HPWL\s+([\d.E+]+),\s*Overflow\s+\[([\d.E+\-,\s]+)\],\s*time\s+([\d.]+)ms'
        
        matches = re.findall(iter_pattern, content)
        
        for match in matches:
            try:
                iter_num = int(match[0])
                hpwl = float(match[1])
                overflow_str = match[2]
                time_ms = float(match[3])
                
                # Parse overflow values
                overflow_values = []
                for val in overflow_str.split(','):
                    val = val.strip()
                    if 'E' in val and not ('E+' in val or 'E-' in val):
                        val = val + '+00'
                    try:
                        overflow_values.append(float(val))
                    except ValueError:
                        overflow_values.append(0.0)
                
                convergence_data.append({
                    'iteration': iter_num,
                    'hpwl': hpwl,
                    'overflow': overflow_values,
                    'time_ms': time_ms,
                    'max_overflow': max(overflow_values) if overflow_values else 0.0,
                    'avg_overflow': sum(overflow_values) / len(overflow_values) if overflow_values else 0.0
                })
                
            except (ValueError, IndexError):
                continue
                
        return convergence_data
        
    def collect_results(self):
        """Collect all results from log files"""
        log_files = list(self.results_dir.glob('**/*.log'))
        
        for log_file in log_files:
            result = self.parse_log_file(log_file)
            if result:
                self.data.append(result)
                
        if not self.data:
            print("No log files found. Looking for alternative data sources...")
            # If no log files, try to infer from directory structure
            self._collect_from_directories()
            
    def _collect_from_directories(self):
        """Fallback: collect basic info from directory structure"""
        subdirs = [d for d in self.results_dir.iterdir() if d.is_dir()]
        
        for subdir in subdirs:
            result = {
                'benchmark': subdir.name,
                'gift_enabled': '_no_gift' not in subdir.name,
                'total_time': None,
                'placement_time': None,
                'final_hpwl': None,
                'final_overflow': [],
                'iterations': None,
                'gift_time': None,
                'reading_time': None
            }
            self.data.append(result)
            
    def create_comparison_df(self):
        """Create a DataFrame for easy comparison"""
        df = pd.DataFrame(self.data)
        
        # Create base benchmark names (without _no_gift suffix)
        df['base_benchmark'] = df['benchmark'].str.replace('_no_gift', '')
        
        # Pivot to have gift and no_gift as columns
        metrics = ['total_time', 'placement_time', 'final_hpwl', 'iterations', 'gift_time']
        
        comparison_data = []
        for base_bench in df['base_benchmark'].unique():
            bench_data = df[df['base_benchmark'] == base_bench]
            
            gift_data = bench_data[bench_data['gift_enabled'] == True]
            no_gift_data = bench_data[bench_data['gift_enabled'] == False]
            
            row = {'benchmark': base_bench}
            
            for metric in metrics:
                gift_val = gift_data[metric].iloc[0] if len(gift_data) > 0 and not gift_data[metric].isna().iloc[0] else None
                no_gift_val = no_gift_data[metric].iloc[0] if len(no_gift_data) > 0 and not no_gift_data[metric].isna().iloc[0] else None
                
                row[f'{metric}_gift'] = gift_val
                row[f'{metric}_no_gift'] = no_gift_val
                
                if gift_val is not None and no_gift_val is not None:
                    row[f'{metric}_improvement'] = ((no_gift_val - gift_val) / no_gift_val) * 100
                else:
                    row[f'{metric}_improvement'] = None
                    
            comparison_data.append(row)
            
        return pd.DataFrame(comparison_data)
        
    def generate_report(self):
        """Generate comprehensive analysis report"""
        if not self.data:
            print("No data available for analysis.")
            return
            
        df = pd.DataFrame(self.data)
        comparison_df = self.create_comparison_df()
        
        print("="*80)
        print("FPGA PLACEMENT RESULTS ANALYSIS")
        print("="*80)
        
        print(f"\nTotal benchmarks analyzed: {len(df)}")
        print(f"GiFT-enabled runs: {len(df[df['gift_enabled'] == True])}")
        print(f"Non-GiFT runs: {len(df[df['gift_enabled'] == False])}")
        
        # Summary statistics
        print("\n" + "="*50)
        print("SUMMARY STATISTICS")
        print("="*50)
        
        valid_comparisons = comparison_df.dropna(subset=['total_time_improvement'])
        
        if len(valid_comparisons) > 0:
            print(f"\nRuntime Analysis (GiFT vs No-GiFT):")
            print(f"Average runtime improvement: {valid_comparisons['total_time_improvement'].mean():.2f}%")
            print(f"Best runtime improvement: {valid_comparisons['total_time_improvement'].max():.2f}%")
            print(f"Worst runtime performance: {valid_comparisons['total_time_improvement'].min():.2f}%")
            
        valid_hpwl = comparison_df.dropna(subset=['final_hpwl_improvement'])
        if len(valid_hpwl) > 0:
            print(f"\nHPWL Quality Analysis:")
            print(f"Average HPWL improvement: {valid_hpwl['final_hpwl_improvement'].mean():.2f}%")
            print(f"Best HPWL improvement: {valid_hpwl['final_hpwl_improvement'].max():.2f}%")
            print(f"Worst HPWL performance: {valid_hpwl['final_hpwl_improvement'].min():.2f}%")
            
        # Iteration analysis
        print(f"\n" + "="*50)
        print("ITERATION ANALYSIS")
        print("="*50)
        
        gift_runs = df[df['gift_enabled'] == True]
        no_gift_runs = df[df['gift_enabled'] == False]
        
        gift_iters = gift_runs['iterations'].dropna()
        no_gift_iters = no_gift_runs['iterations'].dropna()
        
        if len(gift_iters) > 0 and len(no_gift_iters) > 0:
            print(f"Average iterations (GiFT): {gift_iters.mean():.1f}")
            print(f"Average iterations (No-GiFT): {no_gift_iters.mean():.1f}")
            print(f"Iteration difference: {gift_iters.mean() - no_gift_iters.mean():.1f}")
            
        # Convergence analysis
        self._analyze_convergence_patterns(df)
            
        # Detailed comparison table
        print("\n" + "="*50)
        print("DETAILED BENCHMARK COMPARISON")
        print("="*50)
        
        if len(comparison_df) > 0:
            display_cols = ['benchmark', 'total_time_gift', 'total_time_no_gift', 'total_time_improvement',
                           'final_hpwl_gift', 'final_hpwl_no_gift', 'final_hpwl_improvement']
            
            available_cols = [col for col in display_cols if col in comparison_df.columns]
            print(comparison_df[available_cols].to_string(index=False, float_format='%.3f'))
            
        self._generate_plots(df, comparison_df)
        
    def _analyze_convergence_patterns(self, df):
        """Analyze convergence patterns between GiFT and no-GiFT runs"""
        print(f"\nConvergence Pattern Analysis:")
        
        for _, row in df.iterrows():
            if row['convergence_data'] and len(row['convergence_data']) > 10:
                conv_data = row['convergence_data']
                
                # Calculate convergence metrics
                initial_hpwl = conv_data[0]['hpwl']
                final_hpwl = conv_data[-1]['hpwl']
                improvement_rate = (initial_hpwl - final_hpwl) / initial_hpwl * 100
                
                # Find when overflow becomes acceptable (< 0.1)
                convergence_iter = None
                for i, data in enumerate(conv_data):
                    if data['max_overflow'] < 0.1:
                        convergence_iter = i
                        break
                        
                gift_status = "GiFT" if row['gift_enabled'] else "No-GiFT"
                print(f"{row['benchmark']} ({gift_status}):")
                print(f"  - HPWL improvement: {improvement_rate:.1f}%")
                print(f"  - Convergence at iteration: {convergence_iter if convergence_iter else 'Not converged'}")
                print(f"  - Final max overflow: {conv_data[-1]['max_overflow']:.3f}")
        
    def _generate_plots(self, df, comparison_df):
        """Generate visualization plots including iteration analysis"""
        try:
            plt.style.use('seaborn-v0_8')
        except:
            plt.style.use('default')
            
        # Create main comparison plots
        fig1, axes1 = plt.subplots(2, 2, figsize=(15, 12))
        fig1.suptitle('FPGA Placement Results: GiFT vs No-GiFT Comparison', fontsize=16)
        
        # Plot 1: Runtime comparison
        valid_time = comparison_df.dropna(subset=['total_time_gift', 'total_time_no_gift'])
        if len(valid_time) > 0:
            x = np.arange(len(valid_time))
            width = 0.35
            
            axes1[0,0].bar(x - width/2, valid_time['total_time_gift'], width, 
                         label='GiFT', alpha=0.8, color='blue')
            axes1[0,0].bar(x + width/2, valid_time['total_time_no_gift'], width, 
                         label='No-GiFT', alpha=0.8, color='red')
            
            axes1[0,0].set_xlabel('Benchmark')
            axes1[0,0].set_ylabel('Runtime (seconds)')
            axes1[0,0].set_title('Runtime Comparison')
            axes1[0,0].set_xticks(x)
            axes1[0,0].set_xticklabels(valid_time['benchmark'], rotation=45)
            axes1[0,0].legend()
            axes1[0,0].grid(True, alpha=0.3)
            
        # Plot 2: Iteration comparison
        valid_iter = comparison_df.dropna(subset=['iterations_gift', 'iterations_no_gift'])
        if len(valid_iter) > 0:
            x = np.arange(len(valid_iter))
            
            axes1[0,1].bar(x - width/2, valid_iter['iterations_gift'], width, 
                         label='GiFT', alpha=0.8, color='blue')
            axes1[0,1].bar(x + width/2, valid_iter['iterations_no_gift'], width, 
                         label='No-GiFT', alpha=0.8, color='red')
            
            axes1[0,1].set_xlabel('Benchmark')
            axes1[0,1].set_ylabel('Iterations')
            axes1[0,1].set_title('Iteration Count Comparison')
            axes1[0,1].set_xticks(x)
            axes1[0,1].set_xticklabels(valid_iter['benchmark'], rotation=45)
            axes1[0,1].legend()
            axes1[0,1].grid(True, alpha=0.3)
            
        # Plot 3: HPWL comparison
        valid_hpwl = comparison_df.dropna(subset=['final_hpwl_gift', 'final_hpwl_no_gift'])
        if len(valid_hpwl) > 0:
            x = np.arange(len(valid_hpwl))
            
            axes1[1,0].bar(x - width/2, valid_hpwl['final_hpwl_gift'], width, 
                         label='GiFT', alpha=0.8, color='blue')
            axes1[1,0].bar(x + width/2, valid_hpwl['final_hpwl_no_gift'], width, 
                         label='No-GiFT', alpha=0.8, color='red')
            
            axes1[1,0].set_xlabel('Benchmark')
            axes1[1,0].set_ylabel('HPWL')
            axes1[1,0].set_title('Half-Perimeter Wirelength Comparison')
            axes1[1,0].set_xticks(x)
            axes1[1,0].set_xticklabels(valid_hpwl['benchmark'], rotation=45)
            axes1[1,0].legend()
            axes1[1,0].grid(True, alpha=0.3)
            
        # Plot 4: Improvement percentages
        valid_improvements = comparison_df.dropna(subset=['total_time_improvement'])
        if len(valid_improvements) > 0:
            bars = axes1[1,1].bar(valid_improvements['benchmark'], 
                               valid_improvements['total_time_improvement'], 
                               color=['green' if x > 0 else 'red' for x in valid_improvements['total_time_improvement']])
            
            axes1[1,1].set_xlabel('Benchmark')
            axes1[1,1].set_ylabel('Improvement (%)')
            axes1[1,1].set_title('Runtime Improvement (Positive = GiFT Better)')
            axes1[1,1].tick_params(axis='x', rotation=45)
            axes1[1,1].grid(True, alpha=0.3)
            axes1[1,1].axhline(y=0, color='black', linestyle='-', alpha=0.5)
            
        plt.tight_layout()
        plt.savefig('fpga_results_analysis.png', dpi=300, bbox_inches='tight')
        plt.show()
        
        # Generate convergence plots
        self._plot_convergence_curves(df)
        
    def _plot_convergence_curves(self, df):
        """Plot iteration-by-iteration convergence curves"""
        # Find matching benchmark pairs
        benchmark_pairs = {}
        for _, row in df.iterrows():
            base_name = row['benchmark'].replace('_no_gift', '')
            if base_name not in benchmark_pairs:
                benchmark_pairs[base_name] = {}
            
            key = 'gift' if row['gift_enabled'] else 'no_gift'
            if row['convergence_data'] and len(row['convergence_data']) > 0:
                benchmark_pairs[base_name][key] = row['convergence_data']
        
        # Create convergence plots for benchmarks with both variants
        complete_pairs = {k: v for k, v in benchmark_pairs.items() 
                         if 'gift' in v and 'no_gift' in v}
        
        if not complete_pairs:
            print("No matching benchmark pairs found for convergence analysis")
            return
            
        n_pairs = len(complete_pairs)
        fig, axes = plt.subplots(2, min(3, n_pairs), figsize=(15, 10))
        if n_pairs == 1:
            axes = axes.reshape(2, 1)
        elif n_pairs == 2:
            axes = axes.reshape(2, 2)
            
        fig.suptitle('Convergence Analysis: HPWL and Overflow vs Iterations', fontsize=16)
        
        for i, (bench_name, data) in enumerate(list(complete_pairs.items())[:6]):  # Limit to 6 plots
            col = i % 3
            
            # HPWL convergence
            gift_data = data['gift']
            no_gift_data = data['no_gift']
            
            gift_iters = [d['iteration'] for d in gift_data]
            gift_hpwl = [d['hpwl'] for d in gift_data]
            no_gift_iters = [d['iteration'] for d in no_gift_data]
            no_gift_hpwl = [d['hpwl'] for d in no_gift_data]
            
            axes[0, col].plot(gift_iters, gift_hpwl, 'b-', label='GiFT', linewidth=2)
            axes[0, col].plot(no_gift_iters, no_gift_hpwl, 'r-', label='No-GiFT', linewidth=2)
            axes[0, col].set_xlabel('Iteration')
            axes[0, col].set_ylabel('HPWL')
            axes[0, col].set_title(f'{bench_name} - HPWL Convergence')
            axes[0, col].legend()
            axes[0, col].grid(True, alpha=0.3)
            
            # Overflow convergence
            gift_overflow = [d['max_overflow'] for d in gift_data]
            no_gift_overflow = [d['max_overflow'] for d in no_gift_data]
            
            axes[1, col].plot(gift_iters, gift_overflow, 'b-', label='GiFT', linewidth=2)
            axes[1, col].plot(no_gift_iters, no_gift_overflow, 'r-', label='No-GiFT', linewidth=2)
            axes[1, col].axhline(y=0.1, color='green', linestyle='--', alpha=0.7, label='Target (0.1)')
            axes[1, col].set_xlabel('Iteration')
            axes[1, col].set_ylabel('Max Overflow')
            axes[1, col].set_title(f'{bench_name} - Overflow Convergence')
            axes[1, col].legend()
            axes[1, col].grid(True, alpha=0.3)
            axes[1, col].set_yscale('log')
            
        plt.tight_layout()
        plt.savefig('fpga_convergence_analysis.png', dpi=300, bbox_inches='tight')
        plt.show()
        
        print(f"Convergence plots saved as 'fpga_convergence_analysis.png'")

def main():
    """Main execution function"""
    analyzer = FPGAResultsAnalyzer()
    
    print("Collecting FPGA placement results...")
    analyzer.collect_results()
    
    print("Generating analysis report...")
    analyzer.generate_report()
    
    # Save detailed results to CSV
    if analyzer.data:
        df = pd.DataFrame(analyzer.data)
        comparison_df = analyzer.create_comparison_df()
        
        df.to_csv('fpga_detailed_results.csv', index=False)
        comparison_df.to_csv('fpga_comparison_results.csv', index=False)
        
        print(f"\nDetailed results saved to:")
        print(f"- fpga_detailed_results.csv")
        print(f"- fpga_comparison_results.csv")

if __name__ == "__main__":
    main()