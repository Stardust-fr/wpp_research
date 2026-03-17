import pandas as pd
import torch
import torchvision.models as models
import os
from codecarbon import EmissionsTracker

# 1. SETUP MODELS (Using the new 'weights' parameter to avoid warnings)
print("Loading models...")
weights = models.ResNet18_Weights.DEFAULT
model_std = models.resnet18(weights=weights).eval()

# Create the Eco-Mode (Quantized) version
model_eco = torch.quantization.quantize_dynamic(
    model_std, {torch.nn.Linear}, dtype=torch.qint8
)

# 2. LOAD AND CLEAN KAGGLE DATA
# Ensure 'Turbine_Data.csv' is in your D:\wpp_research folder
df = pd.read_csv("T1.csv")

# Remove rows where ActivePower is missing so we don't get 'nan'
df_clean = df.dropna(subset=['ActivePower'])

# We take 24 readings from a part of the day where the turbine is active
# (Starting from index 100 to avoid the calibration/startup rows)
wind_data = df_clean['ActivePower'].iloc[100:124].values

# SET THRESHOLD: If wind is below 1000kW, we switch to Eco-Mode
# (You can adjust this number to see more/less switching)
THRESHOLD = 1000.0 

# 3. RUN WIND-AWARE SIMULATION
tracker = EmissionsTracker()
tracker.start()

print(f"\n{'Hour':<5} | {'Wind (kW)':<10} | {'AI Strategy':<12} | {'Status'}")
print("-" * 50)

for hour, power in enumerate(wind_data):
    if power < THRESHOLD:
        current_model = model_eco
        strategy = "ECO-MODE"
        status = "Saving Grid"
    else:
        current_model = model_std
        strategy = "STANDARD"
        status = "Green Power"

    print(f"{hour:<5} | {power:<10.2f} | {strategy:<12} | {status}")

    # Simulate AI workload (Using a batch of 16 for better measurement)
    with torch.no_grad():
        batch = torch.randn(16, 3, 224, 224)
        current_model(batch)

emissions = tracker.stop()
print("-" * 50)
print(f"Total Carbon for Wind-Aware AI Day: {emissions:.10f} kg CO2")