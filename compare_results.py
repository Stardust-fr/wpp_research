import matplotlib.pyplot as plt
import pandas as pd

# These are the results from your recent runs (approximate values)
# You can update these with your exact 'Total Carbon' numbers
data = {
    'Strategy': ['Always Standard', 'Always Eco', 'Wind-Aware (Ours)'],
    'Carbon (kg CO2)': [0.000187, 0.000185, 0.000111] 
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