
import torch
import torch.nn as nn
from torchvision.models import efficientnet_b0

MODEL_PATH = "models/best_model-v3.pt"

print("=" * 60)
print("VERIFYING EFFICIENTNET-B0 ARCHITECTURE")
print("=" * 60)

# Create EfficientNet-B0
model = efficientnet_b0(weights=None)

# Replace classifier with 2 output classes
model.classifier[1] = nn.Linear(
    model.classifier[1].in_features,
    2
)

# Load checkpoint
state_dict = torch.load(
    MODEL_PATH,
    map_location="cpu",
    weights_only=True
)

print("\nCheckpoint loaded.")
print("Checking architecture compatibility...\n")

try:
    model.load_state_dict(state_dict, strict=True)

    print("[SUCCESS] Architecture matches!")
    print("[SUCCESS] All model weights loaded.")

    # Test forward pass
    model.eval()

    dummy_input = torch.randn(1, 3, 224, 224)

    with torch.no_grad():
        output = model(dummy_input)

    print("\nForward pass successful!")
    print("Input shape:", dummy_input.shape)
    print("Output shape:", output.shape)
    print("Output:", output)

except Exception as e:
    print("\n[FAILED] Architecture mismatch!")
    print("Error:", e)
