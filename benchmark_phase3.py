import cv2
import torch
from realesrgan import RealESRGANer
from basicsr.archs.rrdbnet_arch import RRDBNet
import time
import json
import os

def run_benchmark():
    print("Starting Phase 3 Benchmark...")
    
    # Measurements
    start_time = time.time()
    
    # Device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if torch.cuda.is_available():
        torch.cuda.reset_peak_memory_stats()
        torch.cuda.empty_cache()

    # Model architecture
    model = RRDBNet(
        num_in_ch=3,
        num_out_ch=3,
        num_feat=64,
        num_block=23,
        num_grow_ch=32,
        scale=4
    )

    # Load ESRGAN
    upsampler = RealESRGANer(
        scale=4,
        model_path="models/RealESRGAN_x4plus.pth",
        model=model,
        tile=512,          # VRAM-safe
        tile_pad=10,
        pre_pad=0,
        half=True,         # FP16
        device=devicepp
    )

    # Load 4K image
    input_filename = "input_4k.jpg"
    if not os.path.exists(input_filename):
        raise FileNotFoundError(f"Input file {input_filename} not found.")
        
    img = cv2.imread(input_filename, cv2.IMREAD_COLOR)
    h, w, c = img.shape
    input_res = f"{w}x{h}"

    # Upscale
    # We want to measure the core processing mostly, but the request implies the whole process.
    # I've started the timer at the beginning.
    output, _ = upsampler.enhance(img, outscale=2)

    h_out, w_out, c_out = output.shape
    output_res = f"{w_out}x{h_out}"

    # Save 8K result (optional for benchmark but good for verification)
    output_filename = "output_8k_benchmark.png"
    cv2.imwrite(output_filename, output)

    end_time = time.time()
    elapsed_time = end_time - start_time

    max_memory = 0
    if torch.cuda.is_available():
        max_memory = torch.cuda.max_memory_allocated() / (1024 * 1024) # MB

    results = {
        "phase": "3",
        "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
        "device": torch.cuda.get_device_name(0) if torch.cuda.is_available() else "CPU",
        "input_file": input_filename,
        "resolution_input": input_res,
        "resolution_output": output_res,
        "time_seconds": round(elapsed_time, 2),
        "max_memory_allocated_mb": round(max_memory, 2)
    }

    output_json = "benchmark_phase3.json"
    with open(output_json, "w") as f:
        json.dump(results, f, indent=4)
    
    print(f"Benchmark complete. Results saved to {output_json}")
    print(json.dumps(results, indent=4))

if __name__ == "__main__":
    run_benchmark()
