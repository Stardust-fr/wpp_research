import pandas as pd
import torch
import torchvision.models as models
from codecarbon import EmissionsTracker
import matplotlib.pyplot as plt

# 1. SETUP
weights = models.ResNet18_Weights.DEFAULT
model_std = models.resnet18(weights=weights).eval()
model_eco = torch.quantization.quantize_dynamic(model_std, {torch.nn.Linear}, dtype=torch.qint8)

df = pd.read_csv("T1.csv").dropna(subset=['ActivePower'])
wind_data = df['ActivePower'].iloc[100:124].values
THRESHOLD = 1000.0

def run_simulation(mode_type):
    tracker = EmissionsTracker(measure_power_secs=1, save_to_file=False)
    tracker.start()
    for power in wind_data:
        # Decision Logic
        if mode_type == "STANDARD":
            current_model = model_std
        elif mode_type == "ECO":
            current_model = model_eco
        else: # WIND-AWARE
            current_model = model_std if power > THRESHOLD else model_eco
        
        with torch.no_grad():
            batch = torch.randn(16, 3, 224, 224)
            current_model(batch)
    return tracker.stop()

# 2. EXECUTE THE THREE SCENARIOS
print("Running 'Always Standard' scenario...")
carbon_std = run_simulation("STANDARD")

print("Running 'Always Eco' scenario...")
carbon_eco = run_simulation("ECO")

print("Running 'Wind-Aware' scenario...")
carbon_aware = run_simulation("WIND-AWARE")

# 3. PLOT AND SAVE ACCURATE GRAPH
results = {
    'Strategy': ['Always Standard', 'Always Eco', 'Wind-Aware (Ours)'],
    'Emissions': [carbon_std, carbon_eco, carbon_aware]
}
res_df = pd.DataFrame(results)
res_df.plot(kind='bar', x='Strategy', y='Emissions', color=['red', 'blue', 'green'], legend=False)
plt.ylabel('kg of CO2')
plt.title('24-Hour Comparison: Standard vs Eco vs Wind-Aware')
plt.savefig('final_research_comparison.png')
print("Final Research Chart Saved!")