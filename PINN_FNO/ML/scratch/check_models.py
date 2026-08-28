import torch
import os

MODEL_DIR = "Models"
paths = [
    os.path.join(MODEL_DIR, "exp2_twolayer_pikan_model.pkl"),
    os.path.join(MODEL_DIR, "exp2_twolayer_pikan_non.pkl")
]

for path in paths:
    print(f"Checking {path}...")
    if not os.path.exists(path):
        print("  File does not exist.")
        continue
    try:
        sd = torch.load(path, map_location='cpu', weights_only=False)
        print(f"  Successfully loaded! Keys: {list(sd.keys())[:5]}...")
    except Exception as e:
        print(f"  Failed to load: {e}")
