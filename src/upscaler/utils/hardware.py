import torch
import psutil
import subprocess
import json
import sys

def get_gpu_info():
    info = {
        "name": "Unknown",
        "vram_total_gb": 0,
        "cuda_cores": 0,
        "compute_capability": (0, 0)
    }
    
    if not torch.cuda.is_available():
        return info
        
    try:
        props = torch.cuda.get_device_properties(0)
        info["name"] = props.name
        info["vram_total_gb"] = props.total_memory / (1024**3)
        info["compute_capability"] = (props.major, props.minor)
        info["cuda_cores"] = props.multi_processor_count * 128 # rough estimate for Ampere/Ada, varies
    except:
        pass
        
    return info

def suggest_settings(gpu_info):
    settings = {
        "batch_size": 1,
        "tile_size": 0,
        "fp16": False,
        "trtexec_workspace": 4096 
    }
    
    vram = gpu_info["vram_total_gb"]
    cc = gpu_info["compute_capability"]
    
    # FP16 decision
    # CC 6.0+ (Pascal) supports FP16, but 7.0+ (Volta/Turing) is where Tensor Cores start helping significantly.
    if cc[0] >= 7:
        settings["fp16"] = True
    
    # Batch size & Tile size heuristics for 4x upscaling
    # Target is to fill VRAM but leave room for system.
    # 4k -> 8k upscaling is HUGE memory.
    # Tile size 0 means full frame.
    
    # 5090 (likely 24GB or 32GB?) 
    # If 24GB:
    if vram >= 30: # 5090 32GB?
        settings["batch_size"] = 4
        settings["tile_size"] = 1024 # Can likely handle large tiles or even full 1080p frame
    elif vram >= 20: # 3090/4090 24GB
        settings["batch_size"] = 2
        settings["tile_size"] = 512
    elif vram >= 10: # 3080/4070
        settings["batch_size"] = 1
        settings["tile_size"] = 256
    else:
        settings["batch_size"] = 1
        settings["tile_size"] = 192
        
    print(f"Detected GPU: {gpu_info['name']} ({vram:.1f} GB VRAM, CC {cc[0]}.{cc[1]})")
    print("\nRecommended Settings:")
    print(json.dumps(settings, indent=4))
    
    return settings

if __name__ == "__main__":
    info = get_gpu_info()
    suggest_settings(info)
