import matplotlib.pyplot as plt
import pandas as pd
import json

# Load benchmark data from recent runs
with open('benchmark_summary.json', 'r') as f:
    benchmark_data = json.load(f)

# Load benchmark CSV to compute eco-mode mean
benchmark_df = pd.read_csv('benchmark_results.csv')
eco_mean = benchmark_df['eco_emissions'].mean()

# Extract mean emissions values
standard_mean = benchmark_data['standard_emissions']['mean_kg']
wind_aware_mean = benchmark_data['wind_aware_emissions']['mean_kg']

data = {
    'Strategy': ['Always Standard', 'Always Eco', 'Wind-Aware (Ours)'],
    'Carbon (kg CO2)': [standard_mean, eco_mean, wind_aware_mean] 
}

df = pd.DataFrame(data)

# Create the Plot
plt.figure(figsize=(10, 6))
colors = ['red', 'blue', 'green']
bars = plt.bar(df['Strategy'], df['Carbon (kg CO2)'], color=colors)

plt.title('Total Carbon Emissions (24-Hour Simulation)', fontsize=14)
plt.ylabel('kg of CO2', fontsize=12)
plt.grid(axis='y', linestyle='--', alpha=0.7)

# Add value labels on top of bars
for bar in bars:
    yval = bar.get_height()
    plt.text(bar.get_x() + bar.get_width()/2, yval, f'{yval:.6f}', va='bottom', ha='center')

plt.savefig('carbon_comparison.png')
print("Graph saved as carbon_comparison.png")