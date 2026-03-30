import torch
import torchvision.models as models
import os
from codecarbon import EmissionsTracker

# 1. Load the Standard Model
model_fp32 = models.resnet18(pretrained=True)
model_fp32.eval()

# 2. CREATE THE ECO-MODE (Dynamic Quantization)
# This rounds the "Linear" layers from Float32 to Int8
model_int8 = torch.quantization.quantize_dynamic(
    model_fp32, 
    {torch.nn.Linear}, 
    dtype=torch.qint8
)

# 3. Compare File Sizes
torch.save(model_int8.state_dict(), "eco_model.pt")
std_size = os.path.getsize("standard_model.pt") / 1e6
eco_size = os.path.getsize("eco_model.pt") / 1e6

print(f"Standard Size: {std_size:.2f} MB")
print(f"Eco-Mode Size: {eco_size:.2f} MB")
print(f"Compression: {((std_size - eco_size) / std_size) * 100:.1f}% smaller")

# 4. Measure Eco-Mode Emissions
tracker = EmissionsTracker()
tracker.start()

print("Running Eco-Mode inference...")
dummy_input = torch.randn(1, 3, 224, 224)
for _ in range(10000):
    with torch.no_grad():
        output = model_int8(dummy_input)

emissions_eco = tracker.stop()
print(f"Eco-Mode Emissions: {emissions_eco:.10f} kg of CO2")