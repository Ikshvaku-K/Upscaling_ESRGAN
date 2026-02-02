import torch

try:
    path = "models/BSRGAN.pth"
    checkpoint = torch.load(path, map_location='cpu')
    print("Keys in checkpoint:", checkpoint.keys())
    if isinstance(checkpoint, dict):
        for k in checkpoint.keys():
            if isinstance(checkpoint[k], dict):
                print(f"Sub-keys in {k}:", list(checkpoint[k].keys())[:5])
except Exception as e:
    print(f"Error loading checkpoint: {e}")
