import cv2
import torch
import numpy as np
import time
import json
import os
from basicsr.archs.rrdbnet_arch import RRDBNet
from basicsr.archs.swinir_arch import SwinIR
from realesrgan import RealESRGANer
from skimage import color

def get_device():
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")

def load_realesrgan(device):
    model = RRDBNet(
        num_in_ch=3,
        num_out_ch=3,
        num_feat=64,
        num_block=23,
        num_grow_ch=32,
        scale=4
    )
    upsampler = RealESRGANer(
        scale=4,
        model_path="models/RealESRGAN_x4plus.pth",
        model=model,
        tile=512,
        tile_pad=10,
        pre_pad=0,
        half=True,
        device=device
    )
    return upsampler

def load_bsrgan(device):
    # BSRGAN uses the same RRDBNet architecture as Real-ESRGAN
    model = RRDBNet(
        num_in_ch=3,
        num_out_ch=3,
        num_feat=64,
        num_block=23,
        num_grow_ch=32,
        scale=4
    )
    # BSRGAN weights path
    model_path = "models/BSRGAN.pth"
    
    # Check if model exists
    if not os.path.exists(model_path):
        raise FileNotFoundError(f"BSRGAN model not found at {model_path}")

    # Manually load weights because BSRGAN.pth is a direct state_dict, not {params: ...}
    state_dict = torch.load(model_path, map_location=device)
    
    # Remap keys common in BSRGAN/ESRGAN old models to BasicSR RRDBNet
    new_state_dict = {}
    for k, v in state_dict.items():
        if "RRDB_trunk" in k:
            new_k = k.replace("RRDB_trunk", "body")
        elif "trunk_conv" in k:
            new_k = k.replace("trunk_conv", "conv_body")
        elif "upconv1" in k:
            new_k = k.replace("upconv1", "conv_up1")
        elif "upconv2" in k:
            new_k = k.replace("upconv2", "conv_up2")
        elif "HRconv" in k:
            new_k = k.replace("HRconv", "conv_hr")
        else:
            new_k = k
            
        # Fix casing for RDB blocks (RDB1 -> rdb1)
        new_k = new_k.replace("RDB", "rdb")
        
        new_state_dict[new_k] = v
        
    model.load_state_dict(new_state_dict, strict=True)
    model.eval()
    model = model.to(device)

    # Start with RealESRGANer wrapper for convenience, passing loaded model
    upsampler = RealESRGANer(
        scale=4,
        model_path=None, # Prevent internal reloading
        model=model,
        tile=512,
        tile_pad=10,
        pre_pad=0,
        half=True if device.type == 'cuda' else False, # CPU half precision acts weird sometimes
        device=device
    )
    return upsampler

def load_swinir(device):
    # SwinIR-M configuration
    model = SwinIR(
        upscale=4,
        in_chans=3,
        img_size=64,
        window_size=8,
        img_range=1.,
        depths=[6, 6, 6, 6, 6, 6],
        embed_dim=180,
        num_heads=[6, 6, 6, 6, 6, 6],
        mlp_ratio=2,
        upsampler='pixelshuffle',
        resi_connection='1conv'
    )
    
    model_path = "models/001_classicalSR_DF2K_s64w8_SwinIR-M_x4.pth"
    pretrained_model = torch.load(model_path)
    if 'params_ema' in pretrained_model:
        param_key_g = 'params_ema'
    elif 'params' in pretrained_model:
        param_key_g = 'params'
    else:
        param_key_g = 'params_ema' # Fallback
        
    model.load_state_dict(pretrained_model[param_key_g] if param_key_g in pretrained_model else pretrained_model, strict=True)
    model.eval()
    model = model.to(device)
    return model

def swinir_upscale(model, img, device):
    # img is numpy array (H, W, C), BGR, 0-255
    # Normalize to [0, 1] and Transpose to (C, H, W) and RGB
    img = img.astype(np.float32) / 255.
    img = torch.from_numpy(np.transpose(img[:, :, [2, 1, 0]], (2, 0, 1))).float()
    img = img.unsqueeze(0).to(device)

    # Pad if needed (SwinIR needs multiple of window_size * 8 usually? or just window_size?)
    # The arch handles flexible sizes by internal padding or windowing?
    # Window size is 8.
    _, _, h, w = img.size()
    window_size = 8
    mod_pad_h = (window_size - h % window_size) % window_size
    mod_pad_w = (window_size - w % window_size) % window_size
    img = torch.nn.functional.pad(img, (0, mod_pad_w, 0, mod_pad_h), 'reflect')

    with torch.no_grad():
        output = model(img)

    # Unpad 
    # Output scale is 4
    scale = 4
    _, _, h_out, w_out = output.size()
    output = output[:, :, 0:h_out - mod_pad_h * scale, 0:w_out - mod_pad_w * scale]
    
    # Back to numpy, BGR, 0-255
    output = output.data.squeeze().float().cpu().clamp_(0, 1).numpy()
    output = np.transpose(output[[2, 1, 0], :, :], (1, 2, 0)) # RGB to BGR and CHW to HWC
    output = (output * 255.0).round().astype(np.uint8)
    return output

def compute_color_drift(img1, img2):
    # Convert BGR to LAB
    img1_lab = color.rgb2lab(img1[:, :, ::-1])
    img2_lab = color.rgb2lab(img2[:, :, ::-1])
    
    # Compute Delta E (Euclidean distance in LAB space)
    delta_e = np.sqrt(np.sum((img1_lab - img2_lab)**2, axis=2))
    return np.mean(delta_e)

def generate_diff_map(img1, img2):
    # Absolute difference, amplified
    diff = cv2.absdiff(img1, img2)
    diff = diff * 5 # Amplify differences
    return diff

def log_vram_usage(tag=""):
    if torch.cuda.is_available():
        allocated = torch.cuda.memory_allocated() / (1024 * 1024)
        reserved = torch.cuda.memory_reserved() / (1024 * 1024)
        max_allocated = torch.cuda.max_memory_allocated() / (1024 * 1024)
        print(f"[{tag}] VRAM: Alloc={allocated:.2f}MB, Rsrv={reserved:.2f}MB, MaxAlloc={max_allocated:.2f}MB")

def test_tile_padding(crop_img, device):
    print("Running Tile Padding Test...")
    # Load model once, re-configure tile_pad dynamically if possible, or reload
    # Re-loading to be safe and simple
    
    # Run with Pad 0
    upsampler_0 = RealESRGANer(
        scale=4,
        model_path="models/RealESRGAN_x4plus.pth",
        model=RRDBNet(num_in_ch=3, num_out_ch=3, scale=4, num_feat=64, num_block=23, num_grow_ch=32),
        tile=256, # Force tiling on small image
        tile_pad=0,
        pre_pad=0,
        half=True,
        device=device
    )
    out_0, _ = upsampler_0.enhance(crop_img, outscale=4)
    del upsampler_0
    
    # Run with Pad 10
    upsampler_10 = RealESRGANer(
        scale=4,
        model_path="models/RealESRGAN_x4plus.pth",
        model=RRDBNet(num_in_ch=3, num_out_ch=3, scale=4, num_feat=64, num_block=23, num_grow_ch=32),
        tile=256, # Force tiling on small image
        tile_pad=10,
        pre_pad=0,
        half=True,
        device=device
    )
    out_10, _ = upsampler_10.enhance(crop_img, outscale=4)
    del upsampler_10
    
    # Diff
    diff = cv2.absdiff(out_0, out_10) * 10 # Amplify
    cv2.imwrite("tile_padding_test_diff.jpg", diff)
    print("Tile padding test complete. See tile_padding_test_diff.jpg")


import argparse

def main():
    parser = argparse.ArgumentParser(description="Compare upscaling models on a given input image.")
    parser.add_argument("input", nargs="?", default="input_4k.jpg", help="Path to input image (default: input_4k.jpg)")
    args = parser.parse_args()

    device = get_device()
    print(f"Using device: {device}")

    # Load image
    input_filename = args.input
    if not os.path.exists(input_filename):
        print(f"Error: {input_filename} not found.")
        return

    full_img = cv2.imread(input_filename, cv2.IMREAD_COLOR)
    h, w, c = full_img.shape
    print(f"Input Resolution: {w}x{h}")

    # Center Crop for Comparison (e.g., 512x512)
    crop_size = 512
    cx, cy = w // 2, h // 2
    x1, y1 = cx - crop_size // 2, cy - crop_size // 2
    x2, y2 = x1 + crop_size, y1 + crop_size
    
    # Ensure crop is within bounds
    x1, y1 = max(0, x1), max(0, y1)
    x2, y2 = min(w, x2), min(h, y2)
    
    crop_img = full_img[y1:y2, x1:x2]
    cv2.imwrite("comparison_input_crop.png", crop_img)
    print(f"Running comparison on {crop_size}x{crop_size} center crop.")

    stats = {}
    
    # Baseline for Color Drift: Resized input
    # Upscale input using bicubic to 4x to match output size
    input_upscaled_bicubic = cv2.resize(crop_img, None, fx=4, fy=4, interpolation=cv2.INTER_CUBIC)

    # --- Real-ESRGAN ---
    print("Running Real-ESRGAN...")
    try:
        if torch.cuda.is_available():
            torch.cuda.reset_peak_memory_stats()
            torch.cuda.empty_cache()
            
        realesrgan_model = load_realesrgan(device)
        start_t = time.time()
        output_realesrgan, _ = realesrgan_model.enhance(crop_img, outscale=4)
        end_t = time.time()
        
        log_vram_usage("Real-ESRGAN Success")
        
        max_mem = 0
        if torch.cuda.is_available():
            max_mem = torch.cuda.max_memory_allocated() / (1024 * 1024) # MB
            
        color_drift = compute_color_drift(input_upscaled_bicubic, output_realesrgan)
        diff_map = generate_diff_map(input_upscaled_bicubic, output_realesrgan)
        cv2.imwrite("diff_realesrgan.jpg", diff_map)
            
        stats["Real-ESRGAN"] = {
            "time": round(end_t - start_t, 3),
            "memory_mb": round(max_mem, 2),
            "color_drift_delta_e": round(color_drift, 3)
        }
        cv2.imwrite("comparison_realesrgan.png", output_realesrgan)
        print(f"Real-ESRGAN finished. Peak Memory: {round(max_mem, 2)} MB. Color Drift: {round(color_drift, 3)}")
        del realesrgan_model
        torch.cuda.empty_cache()

    except torch.cuda.OutOfMemoryError as e:
        print(f"Real-ESRGAN OOM Error: {e}")
        log_vram_usage("Real-ESRGAN OOM")
        stats["Real-ESRGAN"] = {"error": "OOM", "details": str(e)}
        torch.cuda.empty_cache()
    except Exception as e:
        print(f"Real-ESRGAN failed: {e}")
        stats["Real-ESRGAN"] = {"error": str(e)}

    # --- BSRGAN ---
    print("Running BSRGAN...")
    try:
        if torch.cuda.is_available():
            torch.cuda.reset_peak_memory_stats()
            torch.cuda.empty_cache()
            
        bsrgan_model = load_bsrgan(device)
        start_t = time.time()
        output_bsrgan, _ = bsrgan_model.enhance(crop_img, outscale=4)
        end_t = time.time()
        
        log_vram_usage("BSRGAN Success")
        
        max_mem = 0
        if torch.cuda.is_available():
            max_mem = torch.cuda.max_memory_allocated() / (1024 * 1024) # MB
            
        color_drift = compute_color_drift(input_upscaled_bicubic, output_bsrgan)
        diff_map = generate_diff_map(input_upscaled_bicubic, output_bsrgan)
        cv2.imwrite("diff_bsrgan.jpg", diff_map)
            
        stats["BSRGAN"] = {
            "time": round(end_t - start_t, 3),
            "memory_mb": round(max_mem, 2),
            "color_drift_delta_e": round(color_drift, 3)
        }
        cv2.imwrite("comparison_bsrgan.png", output_bsrgan)
        print(f"BSRGAN finished. Peak Memory: {round(max_mem, 2)} MB. Color Drift: {round(color_drift, 3)}")
        del bsrgan_model
        torch.cuda.empty_cache()

    except torch.cuda.OutOfMemoryError as e:
        print(f"BSRGAN OOM Error: {e}")
        log_vram_usage("BSRGAN OOM")
        stats["BSRGAN"] = {"error": "OOM", "details": str(e)}
        torch.cuda.empty_cache()
    except Exception as e:
        print(f"BSRGAN failed: {e}")
        stats["BSRGAN"] = {"error": str(e)}

    # --- SwinIR ---
    print("Running SwinIR...")
    try:
        if torch.cuda.is_available():
            torch.cuda.reset_peak_memory_stats()
            torch.cuda.empty_cache()
            
        swinir_model = load_swinir(device)
        start_t = time.time()
        output_swinir = swinir_upscale(swinir_model, crop_img, device)
        end_t = time.time()
        
        log_vram_usage("SwinIR Success")
        
        max_mem = 0
        if torch.cuda.is_available():
            max_mem = torch.cuda.max_memory_allocated() / (1024 * 1024) # MB
            
        color_drift = compute_color_drift(input_upscaled_bicubic, output_swinir)
        diff_map = generate_diff_map(input_upscaled_bicubic, output_swinir)
        cv2.imwrite("diff_swinir.jpg", diff_map)
            
        stats["SwinIR"] = {
            "time": round(end_t - start_t, 3),
            "memory_mb": round(max_mem, 2),
            "color_drift_delta_e": round(color_drift, 3)
        }
        cv2.imwrite("comparison_swinir.png", output_swinir)
        print(f"SwinIR finished. Peak Memory: {round(max_mem, 2)} MB. Color Drift: {round(color_drift, 3)}")
        del swinir_model
        torch.cuda.empty_cache()

    except torch.cuda.OutOfMemoryError as e:
        print(f"SwinIR OOM Error: {e}")
        log_vram_usage("SwinIR OOM")
        stats["SwinIR"] = {"error": "OOM", "details": str(e)}
        torch.cuda.empty_cache()
    except Exception as e:
        print(f"SwinIR failed: {e}")
        import traceback
        traceback.print_exc()
        stats["SwinIR"] = {"error": str(e)}
        
    # --- Tile Padding Test ---
    test_tile_padding(crop_img, device)

    # Save stats
    with open("model_comparison_stats.json", "w") as f:
        json.dump(stats, f, indent=4)
    print("Stats saved to model_comparison_stats.json")
    
    # Create Side-by-Side Comparison
    if all(k in stats and "error" not in stats[k] for k in ["Real-ESRGAN", "BSRGAN", "SwinIR"]):
        # Labeling (Text on Image)
        def draw_label(img, text, color_drift):
            font = cv2.FONT_HERSHEY_SIMPLEX
            # Increase font scale and thickness slightly for readability if needed
            cv2.putText(img, text, (10, 50), font, 1.5, (0, 0, 255), 3, cv2.LINE_AA)
            cv2.putText(img, f"Drift: {color_drift}", (10, 100), font, 1.0, (255, 0, 0), 2, cv2.LINE_AA)
            return img

        p1 = draw_label(output_realesrgan.copy(), "Real-ESRGAN", stats["Real-ESRGAN"]["color_drift_delta_e"])
        p2 = draw_label(output_bsrgan.copy(), "BSRGAN", stats["BSRGAN"]["color_drift_delta_e"])
        p3 = draw_label(output_swinir.copy(), "SwinIR", stats["SwinIR"]["color_drift_delta_e"])
        
        combined = np.hstack([p1, p2, p3])
        cv2.imwrite("model_comparison_result.jpg", combined)
        print("Comparison image saved to model_comparison_result.jpg")

if __name__ == "__main__":
    main()
