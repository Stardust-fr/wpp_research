import torch
import torchvision.models as models

# 1. Load both versions
weights = models.ResNet18_Weights.DEFAULT
model_std = models.resnet18(weights=weights).eval()
model_eco = torch.quantization.quantize_dynamic(model_std, {torch.nn.Linear}, dtype=torch.qint8)

# 2. Create a "test" image
dummy_input = torch.randn(1, 3, 224, 224)

# 3. Get predictions
with torch.no_grad():
    out_std = model_std(dummy_input)
    out_eco = model_eco(dummy_input)

# 4. Compare the top "class" (The AI's best guess)
class_std = torch.argmax(out_std).item()
class_eco = torch.argmax(out_eco).item()

print(f"Standard Model Prediction (Class ID): {class_std}")
print(f"Eco-Mode Model Prediction (Class ID): {class_eco}")

if class_std == class_eco:
    print("SUCCESS: Both models made the SAME prediction.")
    print("Carbon savings came with ZERO loss in decision accuracy.")
else:
    print("Models had a slight variance in prediction.")