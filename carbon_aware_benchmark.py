import pandas as pd
import torch
import torchvision.models as models
from codecarbon import EmissionsTracker
import matplotlib.pyplot as plt
import time  # For Thermal Normalization

# 1. SETUP
print("Initializing Models and SCADA Data...")
weights = models.ResNet18_Weights.DEFAULT
model_std = models.resnet18(weights=weights).eval()

# Create the Eco-Mode (Quantized) version
model_eco = torch.quantization.quantize_dynamic(
    model_std, {torch.nn.Linear}, dtype=torch.qint8
)

# Load Kaggle SCADA Data
df = pd.read_csv("T1.csv").dropna(subset=['ActivePower'])
wind_data = df['ActivePower'].iloc[100:124].values
THRESHOLD = 1000.0

def run_simulation(mode_type):
    # --- THERMAL NORMALIZATION ---
    print(f"\n[Cooldown] Letting GPU/CPU cool for 15s before {mode_type}...")
    time.sleep(15) 
    
    # Fresh tracker for every run to avoid data bleeding
    tracker = EmissionsTracker(measure_power_secs=1, save_to_file=False, log_level="error")
    
    # --- HARDWARE WARM-UP ---
    # Run 5 dummy inferences to stabilize GPU clock speeds before tracking
    warmup_batch = torch.randn(1, 3, 224, 224)
    for _ in range(5):
        model_std(warmup_batch)

    tracker.start()
    for power in wind_data:
        # Decision Logic & Workload Shedding
        if mode_type == "STANDARD":
            current_model = model_std
            batch_size = 16  # Full Workload
        elif mode_type == "ECO":
            current_model = model_eco
            batch_size = 16  # Full Workload but quantized
        else: # WIND-AWARE STRATEGY
            if power > THRESHOLD:
                current_model = model_std
                batch_size = 16 # Surplus Power: Process Max Data
            else:
                current_model = model_eco
                batch_size = 8  # POWER SAVING: Drop batch size (Workload Shedding)
        
        with torch.no_grad():
            batch = torch.randn(batch_size, 3, 224, 224)
            current_model(batch)
            
    emissions = tracker.stop()
    print(f"Finished {mode_type}. Emissions: {emissions:.10f} kg CO2")
    return emissions

# 2. EXECUTE THE THREE SCENARIOS
carbon_std = run_simulation("STANDARD")
carbon_eco = run_simulation("ECO")
carbon_aware = run_simulation("WIND-AWARE")

# 3. PLOT AND SAVE ACCURATE GRAPH
results = {
    'Strategy': ['Always Standard', 'Always Eco', 'Wind-Aware (Ours)'],
    'Emissions': [carbon_std, carbon_eco, carbon_aware]
}
res_df = pd.DataFrame(results)

# Create the visualization
plt.figure(figsize=(10, 6))
colors = ['#ff4d4d', '#4d79ff', '#2eb82e'] # Red, Blue, Green
bars = plt.bar(res_df['Strategy'], res_df['Emissions'], color=colors)

# Add value labels for professional look
for bar in bars:
    yval = bar.get_height()
    plt.text(bar.get_x() + bar.get_width()/2, yval, f'{yval:.8f}', va='bottom', ha='center', fontsize=10)

plt.ylabel('Total kg of CO2 Emissions')
plt.title('WAG-AI: Carbon-Aware Deployment Benchmark')
plt.grid(axis='y', linestyle='--', alpha=0.6)
plt.savefig('final_research_comparison.png')

print("\n" + "="*30)
print(f"SUCCESS: Final Research Chart Saved!")
print(f"Wind-Aware Savings vs Standard: {((carbon_std - carbon_aware)/carbon_std)*100:.2f}%")
print("="*30)