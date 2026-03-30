"""
Multi-Run Carbon-Aware Benchmark Suite
Executes carbon_aware_benchmark.py multiple times and aggregates results
with statistical analysis and visualization.
"""

import subprocess
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import re
import time
from datetime import datetime
import json

# Configuration
NUM_RUNS = 5
OUTPUT_CSV = "benchmark_results.csv"
OUTPUT_JSON = "benchmark_summary.json"
GRAPH_DIR = "graphs/"

# Ensure graphs directory exists
import os
os.makedirs(GRAPH_DIR, exist_ok=True)

def extract_emissions_from_output(output_text):
    """
    Extract emissions values from carbon_aware_benchmark.py output.
    Looks for lines like: "Finished STANDARD. Emissions: 0.0022 kg CO2"
    """
    emissions_dict = {}
    patterns = {
        'STANDARD': r'Finished STANDARD\. Emissions: ([\d.e-]+)',
        'ECO': r'Finished ECO\. Emissions: ([\d.e-]+)',
        'WIND-AWARE': r'Finished WIND-AWARE\. Emissions: ([\d.e-]+)'
    }
    
    for strategy, pattern in patterns.items():
        match = re.search(pattern, output_text)
        if match:
            emissions_dict[strategy] = float(match.group(1))
    
    return emissions_dict

def run_benchmark():
    """Execute a single benchmark run and return emissions data."""
    print(f"\n{'='*60}")
    print(f"Starting benchmark run...")
    print(f"{'='*60}")
    
    try:
        result = subprocess.run(
            ['.venv\\Scripts\\python.exe', 'carbon_aware_benchmark.py'],
            cwd='d:\\wpp_research',
            capture_output=True,
            text=True,
            timeout=600  # 10 minute timeout
        )
        
        print(result.stdout)
        if result.stderr:
            print("STDERR:", result.stderr)
        
        emissions = extract_emissions_from_output(result.stdout)
        
        if not emissions or 'STANDARD' not in emissions:
            print("⚠️ Failed to extract emissions data from this run")
            return None
        
        # Calculate carbon reduction percentage
        carbon_reduction = (
            (emissions['STANDARD'] - emissions['WIND-AWARE']) / emissions['STANDARD']
        ) * 100
        
        return {
            'standard_emissions': emissions['STANDARD'],
            'eco_emissions': emissions.get('ECO', None),
            'wind_aware_emissions': emissions['WIND-AWARE'],
            'carbon_reduction_pct': carbon_reduction,
            'timestamp': datetime.now().isoformat()
        }
    
    except subprocess.TimeoutExpired:
        print("❌ Benchmark run timed out!")
        return None
    except Exception as e:
        print(f"❌ Error running benchmark: {e}")
        return None

def main():
    """Main orchestration function."""
    print(f"\n{'*'*60}")
    print(f"WAG-AI MULTI-RUN BENCHMARK SUITE")
    print(f"Running {NUM_RUNS} iterations of carbon_aware_benchmark.py")
    print(f"{'*'*60}")
    
    results = []
    successful_runs = 0
    
    # Execute multiple runs
    for run_num in range(1, NUM_RUNS + 1):
        print(f"\n[RUN {run_num}/{NUM_RUNS}] Starting at {datetime.now().strftime('%H:%M:%S')}")
        
        emissions_data = run_benchmark()
        
        if emissions_data:
            results.append(emissions_data)
            successful_runs += 1
            print(f"✅ Run {run_num} SUCCESS - Reduction: {emissions_data['carbon_reduction_pct']:.2f}%")
        else:
            print(f"❌ Run {run_num} FAILED")
        
        # Wait between runs for system stabilization
        if run_num < NUM_RUNS:
            print(f"Waiting 20 seconds before next run...")
            time.sleep(20)
    
    print(f"\n{'='*60}")
    print(f"BENCHMARK EXECUTION COMPLETE")
    print(f"Successful runs: {successful_runs}/{NUM_RUNS}")
    print(f"{'='*60}")
    
    if not results:
        print("❌ No successful runs. Exiting.")
        return
    
    # Convert to DataFrame
    df = pd.DataFrame(results)
    
    # Calculate statistics
    stats = {
        'num_runs': successful_runs,
        'carbon_reduction': {
            'mean_pct': df['carbon_reduction_pct'].mean(),
            'std_pct': df['carbon_reduction_pct'].std(),
            'min_pct': df['carbon_reduction_pct'].min(),
            'max_pct': df['carbon_reduction_pct'].max()
        },
        'wind_aware_emissions': {
            'mean_kg': df['wind_aware_emissions'].mean(),
            'std_kg': df['wind_aware_emissions'].std(),
            'min_kg': df['wind_aware_emissions'].min(),
            'max_kg': df['wind_aware_emissions'].max()
        },
        'standard_emissions': {
            'mean_kg': df['standard_emissions'].mean(),
            'std_kg': df['standard_emissions'].std()
        },
        'timestamp': datetime.now().isoformat()
    }
    
    # ========== SAVE RESULTS ==========
    
    # Save detailed results to CSV
    df.to_csv(OUTPUT_CSV, index=False)
    print(f"\n✅ Detailed results saved to: {OUTPUT_CSV}")
    
    # Save statistics summary to JSON
    with open(OUTPUT_JSON, 'w') as f:
        json.dump(stats, f, indent=2)
    print(f"✅ Statistics summary saved to: {OUTPUT_JSON}")
    
    # ========== PRINT STATISTICS ==========
    
    print(f"\n{'*'*60}")
    print(f"STATISTICAL ANALYSIS")
    print(f"{'*'*60}")
    print(f"\nCarbon Reduction (%):")
    print(f"  Mean:              {stats['carbon_reduction']['mean_pct']:.2f}%")
    print(f"  Std Dev:           ±{stats['carbon_reduction']['std_pct']:.2f}%")
    print(f"  Range:             {stats['carbon_reduction']['min_pct']:.2f}% - {stats['carbon_reduction']['max_pct']:.2f}%")
    
    print(f"\nWind-Aware Emissions (kg CO₂):")
    print(f"  Mean:              {stats['wind_aware_emissions']['mean_kg']:.8f}")
    print(f"  Std Dev:           ±{stats['wind_aware_emissions']['std_kg']:.8f}")
    print(f"  Range:             {stats['wind_aware_emissions']['min_kg']:.8f} - {stats['wind_aware_emissions']['max_kg']:.8f}")
    
    print(f"\nStandard (FP32) Emissions (kg CO₂):")
    print(f"  Mean:              {stats['standard_emissions']['mean_kg']:.8f}")
    print(f"  Std Dev:           ±{stats['standard_emissions']['std_kg']:.8f}")
    
    # ========== CREATE VISUALIZATIONS ==========
    
    # Graph 1: Carbon Reduction Over Runs
    fig, ax = plt.subplots(figsize=(10, 6))
    runs = range(1, len(df) + 1)
    ax.plot(runs, df['carbon_reduction_pct'], marker='o', linewidth=2, markersize=8, color='#2ecc71')
    ax.axhline(y=stats['carbon_reduction']['mean_pct'], color='red', linestyle='--', linewidth=2, label=f"Mean: {stats['carbon_reduction']['mean_pct']:.2f}%")
    ax.fill_between(
        runs,
        stats['carbon_reduction']['mean_pct'] - stats['carbon_reduction']['std_pct'],
        stats['carbon_reduction']['mean_pct'] + stats['carbon_reduction']['std_pct'],
        alpha=0.2,
        color='red',
        label=f"±1 Std Dev"
    )
    ax.set_xlabel('Run Number', fontsize=12)
    ax.set_ylabel('Carbon Reduction (%)', fontsize=12)
    ax.set_title('WAG-AI Carbon Reduction Across Multiple Runs', fontsize=14, fontweight='bold')
    ax.grid(True, alpha=0.3)
    ax.legend()
    plt.tight_layout()
    plt.savefig(f'{GRAPH_DIR}01_reduction_over_runs.png', dpi=300)
    print(f"\n✅ Graph saved: 01_reduction_over_runs.png")
    plt.close()
    
    # Graph 2: Emissions Comparison (Box Plot)
    fig, ax = plt.subplots(figsize=(10, 6))
    emissions_data = [
        df['standard_emissions'],
        df['wind_aware_emissions']
    ]
    bp = ax.boxplot(emissions_data, labels=['Standard (FP32)', 'Wind-Aware (Dynamic)'], patch_artist=True)
    
    # Color the boxes
    colors = ['#ff4d4d', '#2ecc71']
    for patch, color in zip(bp['boxes'], colors):
        patch.set_facecolor(color)
        patch.set_alpha(0.7)
    
    ax.set_ylabel('Emissions (kg CO₂)', fontsize=12)
    ax.set_title('Emissions Distribution: Standard vs Wind-Aware', fontsize=14, fontweight='bold')
    ax.grid(True, alpha=0.3, axis='y')
    plt.tight_layout()
    plt.savefig(f'{GRAPH_DIR}02_emissions_boxplot.png', dpi=300)
    print(f"✅ Graph saved: 02_emissions_boxplot.png")
    plt.close()
    
    # Graph 3: Bar Chart - Mean Emissions with Error Bars
    fig, ax = plt.subplots(figsize=(10, 6))
    strategies = ['Standard\n(FP32)', 'Wind-Aware\n(Dynamic)']
    means = [
        stats['standard_emissions']['mean_kg'],
        stats['wind_aware_emissions']['mean_kg']
    ]
    stds = [
        stats['standard_emissions']['std_kg'],
        stats['wind_aware_emissions']['std_kg']
    ]
    colors = ['#ff4d4d', '#2ecc71']
    
    bars = ax.bar(strategies, means, yerr=stds, capsize=10, color=colors, alpha=0.7, edgecolor='black', linewidth=2)
    
    # Add value labels
    for i, (mean, std) in enumerate(zip(means, stds)):
        ax.text(i, mean + std + 0.00001, f'{mean:.6f}\n±{std:.6f}', ha='center', va='bottom', fontsize=10, fontweight='bold')
    
    # Add reduction percentage annotation
    reduction_pct = stats['carbon_reduction']['mean_pct']
    ax.text(0.5, max(means)*0.5, f'{reduction_pct:.1f}%\nReduction', 
            ha='center', va='center', fontsize=16, fontweight='bold', 
            bbox=dict(boxstyle='round', facecolor='yellow', alpha=0.7))
    
    ax.set_ylabel('Total Emissions (kg CO₂)', fontsize=12)
    ax.set_title('Mean Emissions Comparison with Uncertainty (n={})'.format(successful_runs), fontsize=14, fontweight='bold')
    ax.grid(True, alpha=0.3, axis='y')
    plt.tight_layout()
    plt.savefig(f'{GRAPH_DIR}03_emissions_comparison_bars.png', dpi=300)
    print(f"✅ Graph saved: 03_emissions_comparison_bars.png")
    plt.close()
    
    # Graph 4: Distribution Histogram
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    
    # Histogram of carbon reduction percentages
    axes[0].hist(df['carbon_reduction_pct'], bins=5, color='#3498db', alpha=0.7, edgecolor='black')
    axes[0].axvline(stats['carbon_reduction']['mean_pct'], color='red', linestyle='--', linewidth=2, label=f"Mean: {stats['carbon_reduction']['mean_pct']:.2f}%")
    axes[0].set_xlabel('Carbon Reduction (%)', fontsize=11)
    axes[0].set_ylabel('Frequency', fontsize=11)
    axes[0].set_title('Distribution of Carbon Reduction', fontsize=12, fontweight='bold')
    axes[0].legend()
    axes[0].grid(True, alpha=0.3, axis='y')
    
    # Histogram of wind-aware emissions
    axes[1].hist(df['wind_aware_emissions'], bins=5, color='#2ecc71', alpha=0.7, edgecolor='black')
    axes[1].axvline(stats['wind_aware_emissions']['mean_kg'], color='red', linestyle='--', linewidth=2, label=f"Mean: {stats['wind_aware_emissions']['mean_kg']:.6f}")
    axes[1].set_xlabel('Emissions (kg CO₂)', fontsize=11)
    axes[1].set_ylabel('Frequency', fontsize=11)
    axes[1].set_title('Distribution of Wind-Aware Emissions', fontsize=12, fontweight='bold')
    axes[1].legend()
    axes[1].grid(True, alpha=0.3, axis='y')
    
    plt.tight_layout()
    plt.savefig(f'{GRAPH_DIR}04_distributions_histogram.png', dpi=300)
    print(f"✅ Graph saved: 04_distributions_histogram.png")
    plt.close()
    
    # Graph 5: Summary Statistics Table (as image)
    fig, ax = plt.subplots(figsize=(12, 6))
    ax.axis('off')
    
    summary_data = [
        ['Metric', 'Mean', 'Std Dev', 'Min', 'Max'],
        ['Carbon Reduction (%)', 
         f"{stats['carbon_reduction']['mean_pct']:.2f}%",
         f"±{stats['carbon_reduction']['std_pct']:.2f}%",
         f"{stats['carbon_reduction']['min_pct']:.2f}%",
         f"{stats['carbon_reduction']['max_pct']:.2f}%"],
        ['Wind-Aware Emissions (kg CO₂)',
         f"{stats['wind_aware_emissions']['mean_kg']:.8f}",
         f"±{stats['wind_aware_emissions']['std_kg']:.8f}",
         f"{stats['wind_aware_emissions']['min_kg']:.8f}",
         f"{stats['wind_aware_emissions']['max_kg']:.8f}"],
        ['Standard Emissions (kg CO₂)',
         f"{stats['standard_emissions']['mean_kg']:.8f}",
         f"±{stats['standard_emissions']['std_kg']:.8f}",
         'N/A',
         'N/A']
    ]
    
    table = ax.table(cellText=summary_data, cellLoc='center', loc='center', 
                     colWidths=[0.3, 0.15, 0.15, 0.15, 0.15])
    table.auto_set_font_size(False)
    table.set_fontsize(11)
    table.scale(1, 2.5)
    
    # Style header row
    for i in range(5):
        table[(0, i)].set_facecolor('#34495e')
        table[(0, i)].set_text_props(weight='bold', color='white')
    
    # Style data rows
    colors_table = ['#ecf0f1', '#e8f8f5', '#fef5e7']
    for i in range(1, 4):
        for j in range(5):
            table[(i, j)].set_facecolor(colors_table[i-1])
    
    plt.title('Statistical Summary of Multi-Run Benchmark (n={})'.format(successful_runs), 
              fontsize=14, fontweight='bold', pad=20)
    plt.tight_layout()
    plt.savefig(f'{GRAPH_DIR}05_summary_statistics.png', dpi=300, bbox_inches='tight')
    print(f"✅ Graph saved: 05_summary_statistics.png")
    plt.close()
    
    # ========== FINAL SUMMARY ==========
    
    print(f"\n{'*'*60}")
    print(f"FINAL SUMMARY FOR PAPER")
    print(f"{'*'*60}")
    print(f"\nUpdated result for research.txt:")
    print(f"  'WAG-AI achieves {stats['carbon_reduction']['mean_pct']:.1f}% ± {stats['carbon_reduction']['std_pct']:.2f}% total carbon reduction (n={successful_runs} trials)'")
    print(f"\nThis represents:")
    print(f"  - Baseline emissions: {stats['standard_emissions']['mean_kg']:.8f} kg CO₂")
    print(f"  - WAG-AI emissions:   {stats['wind_aware_emissions']['mean_kg']:.8f} kg CO₂")
    print(f"\n✅ All analysis files saved to current directory")
    print(f"✅ All graphs saved to: {GRAPH_DIR}")
    print(f"\nFiles created:")
    print(f"  - {OUTPUT_CSV}")
    print(f"  - {OUTPUT_JSON}")
    print(f"  - {GRAPH_DIR}01_reduction_over_runs.png")
    print(f"  - {GRAPH_DIR}02_emissions_boxplot.png")
    print(f"  - {GRAPH_DIR}03_emissions_comparison_bars.png")
    print(f"  - {GRAPH_DIR}04_distributions_histogram.png")
    print(f"  - {GRAPH_DIR}05_summary_statistics.png")

if __name__ == "__main__":
    main()
