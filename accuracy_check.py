import torch
import torchvision.models as models
import torchvision.transforms as transforms
import torchvision.datasets as datasets
from torch.utils.data import DataLoader
import time

print("="*70)
print("ACCURACY CHECK: Standard (FP32) vs Eco-Mode (INT8 Quantized)")
print("="*70)

# 1. SETUP MODELS
print("\n[1/4] Loading models...")
weights = models.ResNet18_Weights.DEFAULT
model_std = models.resnet18(weights=weights).eval()

model_eco = torch.quantization.quantize_dynamic(
    model_std, 
    {torch.nn.Linear}, 
    dtype=torch.qint8
)

# Move to device (GPU if available for faster inference)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model_std = model_std.to(device)
model_eco = model_eco.to(device)
print(f"  Models loaded on device: {device}")

# 2. LOAD REAL DATASET (CIFAR-10)
print("\n[2/4] Loading CIFAR-10 test dataset...")
transform = transforms.Compose([
    transforms.Resize(224),  # Resize to ResNet18 input size
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406],
                         std=[0.229, 0.224, 0.225])  # ImageNet normalization
])

cifar10_test = datasets.CIFAR10(
    root='./data', 
    train=False, 
    download=True,  # Auto-download if not present
    transform=transform
)

# Use full test set for realistic accuracy
test_loader = DataLoader(cifar10_test, batch_size=32, shuffle=False, num_workers=0)
print(f"  Loaded {len(cifar10_test)} test images from CIFAR-10")

# 3. EVALUATE BOTH MODELS
print("\n[3/4] Running inference on both models...")

correct_std = 0
correct_eco = 0
total = 0
mismatches = 0

start_time = time.time()

with torch.no_grad():
    for batch_idx, (images, labels) in enumerate(test_loader):
        images = images.to(device)
        labels = labels.to(device)
        
        # Standard model predictions
        outputs_std = model_std(images)
        _, predicted_std = torch.max(outputs_std, 1)
        
        # Eco-mode model predictions
        outputs_eco = model_eco(images)
        _, predicted_eco = torch.max(outputs_eco, 1)
        
        # Count correct predictions
        correct_std += (predicted_std == labels).sum().item()
        correct_eco += (predicted_eco == labels).sum().item()
        
        # Track where models disagree
        mismatches += (predicted_std != predicted_eco).sum().item()
        
        total += labels.size(0)
        
        if (batch_idx + 1) % 10 == 0:
            print(f"  Progress: {total}/{len(cifar10_test)} images processed")

inference_time = time.time() - start_time

# 4. CALCULATE AND REPORT RESULTS
print("\n[4/4] Computing accuracy metrics...")
print("\n" + "="*70)
print("RESULTS")
print("="*70)

accuracy_std = 100.0 * correct_std / total
accuracy_eco = 100.0 * correct_eco / total
accuracy_loss = accuracy_std - accuracy_eco
agreement_rate = 100.0 * (total - mismatches) / total

print(f"\nStandard Model (FP32):")
print(f"  Accuracy: {accuracy_std:.2f}% ({correct_std}/{total})")

print(f"\nEco-Mode Model (INT8 Quantized):")
print(f"  Accuracy: {accuracy_eco:.2f}% ({correct_eco}/{total})")

print(f"\nAccuracy Loss from Quantization:")
print(f"  Absolute: {accuracy_loss:.2f} percentage points")
print(f"  Relative: {(accuracy_loss / accuracy_std) * 100:.2f}% degradation")

print(f"\nModel Agreement Rate:")
print(f"  Both models agree on {agreement_rate:.2f}% of predictions ({total - mismatches}/{total})")
print(f"  Models disagree on {100 - agreement_rate:.2f}% of predictions ({mismatches}/{total})")

print(f"\nInference Performance:")
print(f"  Total inference time: {inference_time:.2f} seconds")
print(f"  Average time per image: {(inference_time / total) * 1000:.2f} ms")

print("\n" + "="*70)
print("CONCLUSION")
print("="*70)
if accuracy_loss < 1.0:
    print(f"✓ Quantization has MINIMAL impact: {accuracy_loss:.2f}% accuracy loss")
    print("  → Eco-mode is viable for most applications")
elif accuracy_loss < 3.0:
    print(f"⚠ Quantization has MODERATE impact: {accuracy_loss:.2f}% accuracy loss")
    print("  → Use Wind-Aware to maintain accuracy when green energy available")
else:
    print(f"✗ Quantization has SIGNIFICANT impact: {accuracy_loss:.2f}% accuracy loss")
    print("  → Consider different quantization methods")

print("="*70)