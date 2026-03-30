import torch
import torchvision.models as models
import os
from codecarbon import EmissionsTracker

# 1. Load the "Standard" Model
model = models.resnet18(pretrained=True)
model.eval()

# 2. Check the "Weight" of the Model
torch.save(model.state_dict(), "standard_model.pt")
print(f"Standard Model Size: {os.path.getsize('standard_model.pt') / 1e6:.2f} MB")

# 3. Track Energy Usage for 100 Inferences
tracker = EmissionsTracker()
tracker.start()

print("Running baseline inference...")
dummy_input = torch.randn(1, 3, 224, 224)
for _ in range(10000):
    with torch.no_grad():
        output = model(dummy_input)

emissions = tracker.stop()
print(f"Baseline Emissions: {emissions:.10f} kg of CO2")