
import os
import torch

MODEL_PATH = "models/best_model-v3.pt"

print("=" * 60)
print("DEEPFAKE DETECTOR - MODEL CHECK")
print("=" * 60)

# 1. Check whether file exists
if not os.path.exists(MODEL_PATH):
    print("[ERROR] Model file not found!")
    print("Expected location:", os.path.abspath(MODEL_PATH))
    raise SystemExit(1)

print("[SUCCESS] Model file exists!")

# 2. Check file size
size_mb = os.path.getsize(MODEL_PATH) / (1024 * 1024)
print(f"Model size: {size_mb:.2f} MB")

if size_mb == 0:
    print("[ERROR] Model file is empty!")
    raise SystemExit(1)

# 3. Load model checkpoint
try:
    checkpoint = torch.load(
        MODEL_PATH,
        map_location="cpu",
        weights_only=True
    )

    print("[SUCCESS] Checkpoint loaded successfully!")

except Exception as e:
    print("[ERROR] Could not load checkpoint:")
    print(e)
    raise SystemExit(1)

# 4. Inspect checkpoint structure
print("\nCheckpoint type:", type(checkpoint))

if isinstance(checkpoint, dict):
    print("\nNumber of entries:", len(checkpoint))

    print("\nFirst 15 keys:")
    for key in list(checkpoint.keys())[:15]:
        value = checkpoint[key]

        if isinstance(value, torch.Tensor):
            print(key, "->", tuple(value.shape))
        else:
            print(key, "->", type(value))

print("\nMODEL FILE CHECK COMPLETED")
